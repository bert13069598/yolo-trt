import cv2
import torch

from predict.base import TRT
from utils.color import colormap
from utils.labels import car_label_to_object, yolo_labels


class YOLOv10_TRT(TRT):
    def __init__(self, model, batch, quantization, nc=None):
        self.model = model
        trt_engine = "models/{}-b{}-{}.engine".format(model, batch, quantization)
        super().__init__(trt_engine, batch, quantization, (3840, 2160), (640, 640))

        self.output_buffer = torch.empty((self.batch, 300, 6),
                                         dtype=self.dtype,
                                         device=self.DEVICE)

    def postprocess(self, *imgs, results, conf_thres=0.25):
        bs = results.shape[0]
        xc = results[..., 4] > conf_thres

        outputs = [torch.zeros((0, 6), device=results.device)] * bs
        for xi, x in enumerate(results):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue

            x[:, 0:4:2] = self.IM[0][0] * x[:, 0:4:2] + self.IM[0][2]
            x[:, 1:4:2] = self.IM[1][1] * x[:, 1:4:2] + self.IM[1][2]
            outputs[xi] = x

        for img, output in zip(imgs, outputs):
            boxes, confs, classes = output.split((4, 1, 1), dim=1)
            confs = confs.squeeze(1).cpu()
            classes = classes.squeeze(1).cpu()

            for i, box in enumerate(boxes):
                confidence = confs[i]
                label = int(classes[i])
                x1, y1, x2, y2 = map(int, box)
                r, g, b = map(int, colormap[label])
                if self.model == 'yolov10n':
                    caption = f"{yolo_labels[label]} {confidence:.2f}"
                elif self.model == 'yolov10n-car':
                    caption = f"{car_label_to_object[label]} {confidence:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 2)
                cv2.putText(img, caption, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 2)

        return imgs
