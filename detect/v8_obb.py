import cv2
import numpy as np
import torch

from detect.base import TRT
from utils.color import colormap
from utils.labels import dota_label_to_object, car_label_to_object


def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i: i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)


def _get_covariance_matrix(boxes):
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2))
          / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
          / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps) + eps).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def NMS(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_idx]

    ious = batch_probiou(boxes, boxes).triu_(diagonal=1)
    pick = torch.nonzero(ious.max(dim=0)[0] < iou_thres).squeeze_(-1)
    return sorted_idx[pick]


class YOLOv8_OBB_TRT(TRT):
    def __init__(self, model, batch, quantization, nc):
        self.model = model
        trt_engine = "models/{}-b{}-{}.engine".format(model, batch, quantization)
        super().__init__(trt_engine, batch, quantization, (3840, 2160), (1024, 1024))

        self.input_buffer = torch.empty((self.batch, 3, self.dst_shape[1], self.dst_shape[0]),
                                        dtype=self.dtype,
                                        device=self.DEVICE)
        self.output_buffer = torch.empty((self.batch, nc + 5, 21504),
                                         dtype=self.dtype,
                                         device=self.DEVICE)

    def postprocess(self, *imgs, results, conf_thres=0.25, iou_thres=0.45):
        bs = results.shape[0]  # batch size
        nc = results.shape[1] - 5  # num of cls
        xc = results[:, 4:4 + nc].amax(dim=1) > conf_thres  # candidates

        results = results.transpose(-1, -2)

        outputs = [torch.zeros((0, 7), device=results.device)] * bs
        for xi, x in enumerate(results):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            box, cls, angle = x.split((4, nc, 1), dim=1)

            # Best class only
            conf, label = cls.max(1, keepdim=True)
            x = torch.cat((box, angle, conf, label.float()), dim=1)[conf.view(-1) > conf_thres]

            if not x.shape[0]:
                continue

            cx = x[:, 0]
            cy = x[:, 1]
            wh = x[:, 2:4]
            x[:, 0] = self.IM[0][0] * cx + self.IM[0][2]
            x[:, 1] = self.IM[1][1] * cy + self.IM[1][2]
            x[:, 2:4] = self.IM[0][0] * wh

            i = NMS(x[:, :5], x[:, 4], iou_thres)  # xywhr, conf
            outputs[xi] = x[i]

        for img, output in zip(imgs, outputs):
            boxes, confs, classes = output.split((5, 1, 1), dim=1)
            boxes = np.asarray(xywhr2xyxyxyxy(boxes).cpu())
            confs = np.asarray(confs.squeeze(1).cpu())
            classes = np.asarray(classes.squeeze(1).cpu(), dtype=int)

            if len(outputs):
                for i, box in enumerate(boxes):
                    confidence = confs[i]
                    label = classes[i]
                    r, g, b = map(int, colormap[label])
                    cv2.polylines(img, [np.asarray(box, dtype=int)], True, (r, g, b), 2)
                    if self.model == 'yolov8n-obb':
                        caption = f"{dota_label_to_object[label]} {confidence:.2f}"
                    elif self.model == 'yolov8n-obb-car':
                        caption = f"{car_label_to_object[label]} {confidence:.2f}"
                    w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
                    left, top = [int(b) for b in box[0]]
                    cv2.putText(img, caption, (left, top - 5), 0, 1, (r, g, b), 2, 16)
        return imgs
