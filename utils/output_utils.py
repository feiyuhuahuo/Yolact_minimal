import torch.nn.functional as F
import cv2
from utils.box_utils import crop, box_iou, box_iou_numpy, crop_numpy
import torch
import numpy as np
from config import COLORS
from cython_nms import nms as cnms
import pdb


def fast_nms(box_thre, coef_thre, class_thre, cfg):
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.size()
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms


def fast_nms_numpy(box_thre, coef_thre, class_thre, cfg):
    # descending sort
    idx = np.argsort(-class_thre, axis=1)
    class_thre = np.sort(class_thre, axis=1)[:, ::-1]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.shape
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou_numpy(box_thre, box_thre)
    iou = np.triu(iou, k=1)
    iou_max = np.max(iou, axis=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = np.tile(np.arange(num_classes)[:, None], (1, keep.shape[1]))

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    idx = np.argsort(-class_nms, axis=0)
    class_nms = np.sort(class_nms, axis=0)[::-1]

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms


def traditional_nms(boxes, masks, scores, cfg):
    num_classes = scores.size(0)

    idx_lst, cls_lst, scr_lst = [], [], []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * cfg.img_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > cfg.nms_score_thre
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, cfg.nms_iou_thre)
        keep = torch.tensor(keep, device=boxes.device).long()

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


def nms(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]

    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes
    class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

    # decode boxes
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        if not cfg.traditional_nms:
            box_thre, coef_thre, class_ids, class_thre = fast_nms(box_thre, coef_thre, class_thre, cfg)
        else:
            box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, coef_thre, proto_p


def nms_numpy(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]

    class_p = class_p.transpose(1, 0)
    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes

    class_p_max = np.max(class_p, axis=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]

    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

    # decode boxes
    box_thre = np.concatenate((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                               anchor_thre[:, 2:] * np.exp(box_thre[:, 2:] * 0.2)), axis=1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        assert not cfg.traditional_nms, 'Traditional nms is not supported with numpy.'
        box_thre, coef_thre, class_ids, class_thre = fast_nms_numpy(box_thre, coef_thre, class_thre, cfg)
        return class_ids, class_thre, box_thre, coef_thre, proto_p


def after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None, img_name=None):
    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    if cfg and cfg.save_lincomb:
        draw_lincomb(proto_p, coef_p, img_name)

    masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t()))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop(masks, box_p)

    masks = masks.permute(2, 0, 1).contiguous()

    ori_size = max(img_h, img_w)
    # in OpenCV, cv2.resize is `align_corners=False`.
    masks = F.interpolate(masks.unsqueeze(0), (ori_size, ori_size), mode='bilinear', align_corners=False).squeeze(0)
    masks.gt_(0.5)  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.int()

    return ids_p, class_p, box_p, masks


def after_nms_numpy(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None):
    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    assert not cfg.save_lincomb, 'save_lincomb is not supported in onnx mode.'

    masks = np_sigmoid(np.matmul(proto_p, coef_p.T))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop_numpy(masks, box_p)

    ori_size = max(img_h, img_w)
    masks = cv2.resize(masks, (ori_size, ori_size), interpolation=cv2.INTER_LINEAR)

    if masks.ndim == 2:
        masks = masks[:, :, None]

    masks = np.transpose(masks, (2, 0, 1))
    masks = masks > 0.5  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.astype('int32')

    return ids_p, class_p, box_p, masks


def draw_lincomb(proto_data, masks, img_name):
    for kdx in range(1):
        jdx = kdx + 0
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

        arr_img = ((arr_img + 1) * 127.5).astype('uint8')
        arr_img = cv2.applyColorMap(arr_img, cv2.COLORMAP_WINTER)
        cv2.imwrite(f'results/images/lincomb_{img_name}', arr_img)


def draw_img(ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=None):
    if ids_p is None:
        return img_origin

    if isinstance(ids_p, torch.Tensor):
        ids_p = ids_p.cpu().numpy()
        class_p = class_p.cpu().numpy()
        box_p = box_p.cpu().numpy()
        mask_p = mask_p.cpu().numpy()

    num_detected = ids_p.shape[0]

    img_fused = img_origin
    if not cfg.hide_mask:
        masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
        # The color of the overlap area is different because of the '%' operation.
        masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
        color_masks = COLORS[masks_semantic].astype('uint8')
        img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

        if cfg.cutout:
            for i in range(num_detected):
                one_obj = np.tile(mask_p[i], (3, 1, 1)).transpose((1, 2, 0))
                one_obj = one_obj * img_origin
                new_mask = mask_p[i] == 0
                new_mask = np.tile(new_mask * 255, (3, 1, 1)).transpose((1, 2, 0))
                x1, y1, x2, y2 = box_p[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)

    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    if not cfg.hide_bbox:
        for i in reversed(range(num_detected)):
            x1, y1, x2, y2 = box_p[i, :]

            color = COLORS[ids_p[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

            class_name = cfg.class_names[ids_p[i]]
            text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    if cfg.real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused
