import cv2
import numpy as np
import torch
import torchvision

from predict.base import TRT
from utils.color import colormap
from utils.labels import yolo_labels


def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


class YOLOv9_TRT(TRT):
    def __init__(self, model, batch, quantization, nc=None):
        self.model = model
        trt_engine = "models/{}-b{}-{}.engine".format(model, batch, quantization)
        super().__init__(trt_engine, batch, quantization, (3840, 2160), (640, 640))

        self.output_buffer = torch.empty((self.batch, 84, 8400),
                                         dtype=self.dtype,
                                         device=self.DEVICE)

    def postprocess(self, *imgs, results, conf_thres=0.25, iou_thres=0.45):
        bs = results.shape[0]  # batch size
        nc = results.shape[1] - 4  # num of cls
        xc = results[:, 4:].amax(1) > conf_thres

        results = results.transpose(-1, -2)
        results = torch.cat((xywh2xyxy(results[..., :4]), results[..., 4:]), dim=-1)

        outputs = [torch.zeros((0, 6), device=results.device)] * bs
        for xi, x in enumerate(results):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            box, cls = x.split((4, nc), dim=1)

            conf, label = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, label.float()), dim=1)[conf.view(-1) > conf_thres]

            if not x.shape[0]:
                continue

            x[:, 0:4:2] = self.IM[0][0] * x[:, 0:4:2] + self.IM[0][2]
            x[:, 1:4:2] = self.IM[1][1] * x[:, 1:4:2] + self.IM[1][2]

            # Batched NMS
            scores = x[:, 4]  # scores
            boxes = x[:, :4]  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            outputs[xi] = x[i]

        for img, output in zip(imgs, outputs):
            boxes, confs, classes = output.split((4, 1, 1), dim=1)
            confs = confs.squeeze(1).cpu()
            classes = classes.squeeze(1).cpu()

            for i, box in enumerate(boxes):
                confidence = confs[i]
                label = int(classes[i])
                x1, y1, x2, y2 = map(int, box)
                r, g, b = map(int, colormap[label])
                caption = f"{yolo_labels[label]} {confidence:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 2)
                cv2.putText(img, caption, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 2)

        return imgs
