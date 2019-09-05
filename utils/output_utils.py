""" Contains functions used to sanitize and prepare the output of Yolact. """
import torch.nn.functional as F
from data.config import cfg
from utils.box_utils import crop, sanitize_coordinates
import torch
from utils.box_utils import decode, jaccard
from utils import timer
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
from utils.cython_nms import nms as cnms


class NMS(object):
    def __init__(self, top_k, conf_thresh, nms_thresh, trad_nms=False):
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.trad_nms = trad_nms

    def __call__(self, predictions):
        box_p = predictions['box'].squeeze()  # [19248, 4]
        class_p = predictions['class'].squeeze()  # [19248, 81]
        coef_p = predictions['coef'].squeeze()  # [19248, 32]
        priors = predictions['priors']  # [19248, 4]
        proto_p = predictions['proto'].squeeze()  # [138, 138, 32]

        with timer.env('Detect'):
            class_p = class_p.transpose(1, 0).contiguous()  # [81, 19248]
            box_decode = decode(box_p, priors)  # [19248, 4]

            # exclude the background class
            class_p = class_p[1:, :]
            # get the max score class of 19248 predicted boxes
            class_p_max, _ = torch.max(class_p, dim=0)  # [19248]

            # filter predicted boxes according the class score
            keep = (class_p_max > self.conf_thresh)
            class_thre = class_p[:, keep]
            box_thre = box_decode[keep, :]
            coef_thre = coef_p[keep, :]

            if class_thre.size(1) == 0:
                result = None

            else:
                if not self.trad_nms:
                    box_thre, coef_thre, classes, class_thre = self.fast_nms(box_thre, coef_thre, class_thre)
                else:
                    box_thre, coef_thre, classes, class_thre = self.traditional_nms(box_thre, coef_thre, class_thre)

                result = {'box': box_thre, 'mask': coef_thre, 'class': classes, 'score': class_thre}

                if result is not None and proto_p is not None:
                    result['proto'] = proto_p

        return result

    def fast_nms(self, box_thre, coef_thre, class_thre, second_threshold: bool = False):
        class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]
        idx = idx[:, :self.top_k].contiguous()
        class_thre = class_thre[:, :self.top_k]

        num_classes, num_dets = idx.size()

        box_thre = box_thre[idx.view(-1), :].view(num_classes, num_dets, 4)  # [80, 64, 4]
        coef_thre = coef_thre[idx.view(-1), :].view(num_classes, num_dets, -1)  # [80, 64, 32]

        iou = jaccard(box_thre, box_thre)

        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= self.nms_thresh)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (class_thre > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)

        classes = classes[keep]

        box_nms = box_thre[keep]
        coef_nms = coef_thre[keep]
        class_nms = class_thre[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        class_nms, idx = class_nms.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        class_nms = class_nms[:cfg.max_num_detections]

        classes = classes[idx]
        box_nms = box_nms[idx]
        coef_nms = coef_nms[idx]

        '''
        Test code, a little mAP dropped.
        If one box predicts more than one class, only keep the highest score duplicate.
        '''
        # box_list = np.array(box_nms.cpu()).tolist()
        # class_nms_list = np.array(class_nms.cpu()).tolist()
        #
        # repeat = []
        # ss = list(np.arange(len(box_list)))
        #
        # for aa in box_list:
        #     if (box_list.count(aa) > 1) and (aa not in repeat):
        #         repeat.append(aa)
        #
        # for aa in repeat:
        #     id1 = [j for j, bb in enumerate(box_list) if bb == aa]
        #     temp = [class_nms_list[aa] for aa in id1]
        #     temp = np.array(temp).argmax()
        #     id1.remove(id1[temp])
        #
        #     for jj in id1:
        #         ss.remove(jj)
        #
        # box_nms = box_nms[ss]
        # coef_nms = coef_nms[ss]
        # classes = classes[ss]
        # class_nms = class_nms[ss]

        return box_nms, coef_nms, classes, class_nms

    @staticmethod
    def traditional_nms(boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

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
        classes = torch.cat(cls_lst, dim=0)
        scores = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores


def postprocess(dets, w, h, interpolation_mode='bilinear', visualize_lincomb=False,
                crop_masks=True, score_threshold=0, img_name=None):
    
    if dets is None:
        return [torch.Tensor()] * 4  # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    # At this points masks is only the coefficients
    proto_data = dets['proto']

    if visualize_lincomb:
        display_lincomb(proto_data, masks, img_name)

    masks = torch.sigmoid(torch.matmul(proto_data, masks.t()))

    # Crop masks before upsampling because you know why
    if crop_masks:
        masks = crop(masks, boxes)

    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.permute(2, 0, 1).contiguous()
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

    # Binarize the masks
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = boxes.long()

    return classes, scores, boxes, masks


def display_lincomb(proto_data, masks, img_name):
    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        
        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
        arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1/(1+np.exp(-running_total)))

                arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (running_total_nonlin > 0.5).astype(np.float)

        plt.imshow(arr_img)
        plt.savefig(f'results/images/lincomb_{img_name}')
