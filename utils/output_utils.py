import torch.nn.functional as F
import cv2
from utils.box_utils import crop, sanitize_coordinates
import torch
from utils.box_utils import decode, jaccard
from utils import timer
import numpy as np
import pyximport
from data.config import cfg, COLORS

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
from utils.cython_nms import nms as cnms


def fast_nms(box_thre, coef_thre, class_thre, second_threshold: bool = False):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k].contiguous()
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()

    box_thre = box_thre[idx.view(-1), :].view(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.view(-1), :].view(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = jaccard(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_thre)

    # We should also only keep detections over the confidence threshold, but at the cost of
    # maxing out your detection count for every image, you can just not do that. Because we
    # have such a minimal amount of computation per detection (matrix mulitplication only),
    # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
    # However, when you implement this in your method, you should do this second threshold.
    if second_threshold:
        keep *= (class_thre > cfg.conf_thresh)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids = class_ids[keep]

    box_nms = box_thre[keep]
    coef_nms = coef_thre[keep]
    class_nms = class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms


def traditional_nms(boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
    num_classes = scores.size(0)

    idx_lst = []
    cls_lst = []
    scr_lst = []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * cfg.img_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > conf_thresh
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, iou_threshold)
        keep = torch.Tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])

    idx = torch.cat(idx_lst, dim=0)
    class_ids = torch.cat(cls_lst, dim=0)
    scores = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:cfg.max_detections]
    scores = scores[:cfg.max_detections]

    idx = idx[idx2]
    class_ids = class_ids[idx2]

    # Undo the multiplication above
    return boxes[idx] / cfg.img_size, masks[idx], class_ids, scores


def NMS(net_outs, trad_nms=False):
    box_p = net_outs['box'].squeeze()  # [19248, 4]
    class_p = net_outs['class'].squeeze()  # [19248, 81]
    coef_p = net_outs['coef'].squeeze()  # [19248, 32]
    anchors = net_outs['anchors']  # [19248, 4]
    proto_p = net_outs['proto'].squeeze()  # [138, 138, 32]

    with timer.env('Detect'):
        class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]
        box_decode = decode(box_p, anchors)  # [19248, 4]

        # exclude the background class
        class_p = class_p[1:, :]
        # get the max score class of 19248 predicted boxes
        class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

        # filter predicted boxes according the class score
        keep = (class_p_max > cfg.conf_thre)
        class_thre = class_p[:, keep]
        box_thre = box_decode[keep, :]
        coef_thre = coef_p[keep, :]

        if class_thre.size(1) == 0:
            result = None

        else:
            if not trad_nms:
                box_thre, coef_thre, class_ids, class_thre = fast_nms(box_thre, coef_thre, class_thre)
            else:
                box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre)

            result = {'box': box_thre, 'coef': coef_thre, 'class_ids': class_ids, 'class': class_thre}

            if result is not None and proto_p is not None:
                result['proto'] = proto_p

    return result


def after_nms(nms_outs, img_h, img_w, show_lincomb=False, crop_masks=True, visual_thre=0, img_name=None):
    if nms_outs is None:
        return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

    if visual_thre > 0:
        keep = nms_outs['class'] > visual_thre

        for k in nms_outs:
            if k != 'proto':
                nms_outs[k] = nms_outs[k][keep]

        if nms_outs['class'].size(0) == 0:
            return [torch.Tensor()] * 4

    class_ids = nms_outs['class_ids']
    boxes = nms_outs['box']
    classes = nms_outs['class']
    coefs = nms_outs['coef']

    # At this points masks is only the coefficients
    proto_data = nms_outs['proto']

    if show_lincomb:
        draw_lincomb(proto_data, coefs, img_name)

    masks = torch.sigmoid(torch.matmul(proto_data, coefs.t()))

    # Crop masks by boxes
    if crop_masks:
        masks = crop(masks, boxes)

    masks = masks.permute(2, 0, 1).contiguous()
    masks = F.interpolate(masks.unsqueeze(0), (img_h, img_w), mode='bilinear', align_corners=False).squeeze(0)
    # Binarize the masks
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], img_w)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], img_h)
    boxes = boxes.long()

    return class_ids, classes, boxes, masks


def draw_lincomb(proto_data, masks, img_name):
    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        p_h, p_w, _ = proto_data.size()
        arr_img = np.zeros([p_h * arr_h, p_w * arr_w])
        arr_run = np.zeros([p_h * arr_h, p_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1 / (1 + np.exp(-running_total)))

                arr_img[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (proto_data[:, :, idx[i]] / torch.max(
                    proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (running_total_nonlin > 0.5).astype(np.float)

        plt.imshow(arr_img)
        plt.savefig(f'results/images/lincomb_{img_name}')


def draw_img(results, img_origin, img_name, args, fps=None):
    class_ids, classes, boxes, masks = [x.cpu().numpy() for x in results]
    num_detected = class_ids.shape[0]

    if num_detected == 0:
        # No detections found so just output the original image
        return img_origin

    if not args.hide_mask:
        masks_semantic = masks * (class_ids[:, None, None] + 1)  # expand class_ids' shape for broadcasting
        # The color of the overlap area is different because of the '%' operation.
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
        color_masks = COLORS[masks_semantic].astype('uint8')
        img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        if args.cutout:
            for i in range(num_detected):
                one_obj = np.tile(masks[i], (3, 1, 1)).transpose((1, 2, 0))
                one_obj = one_obj * img_origin
                new_mask = masks[i] == 0
                new_mask = np.tile(new_mask * 255, (3, 1, 1)).transpose((1, 2, 0))
                x1, y1, x2, y2 = boxes[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)

    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    if not args.hide_bbox:
        for i in reversed(range(num_detected)):
            x1, y1, x2, y2 = boxes[i, :]

            color = COLORS[class_ids[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

            class_name = cfg.dataset.class_names[class_ids[i]]
            text_str = f'{class_name}: {classes[i]:.2f}' if not args.hide_score else class_name

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if args.real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused
