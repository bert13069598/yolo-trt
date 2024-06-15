import argparse
import threading
from glob import glob

import cv2
import torch

from base import TRT
from utils.color import colormap
from utils.labels import car_label_to_object, yolo_labels

parser = argparse.ArgumentParser(description='YOLOv10')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov10n')
parser.add_argument('-b', '--batch', type=int, help='batch number', default=1)
parser.add_argument('-q', '--quantization', type=str, help='when export, fp32 fp16 int8', default='fp32')
args = parser.parse_args()


class YOLOv10_TRT(TRT):
    def __init__(self, trt_engine):
        super().__init__(trt_engine, args.batch, args.quantization, (3840, 2160), (640, 640))

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
                if args.model == 'yolov10n':
                    caption = f"{yolo_labels[label]} {confidence:.2f}"
                elif args.model == 'yolov10n-car':
                    caption = f"{car_label_to_object[label]} {confidence:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (r, g, b), 2)
                cv2.putText(img, caption, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (r, g, b), 2)

        return imgs


if __name__ == "__main__":
    path = "/home/deepet/Downloads/datasets/자율주행드론 비행 영상/Training/image"
    paths1 = sorted(glob(path + "/202007171220_60m_45도_2_image/*.jpg"))
    paths2 = sorted(glob(path + "/202007171454_60m_45도_2_image/*.jpg"))
    paths3 = sorted(glob(path + "/202007171527_80m_45도_2_image/*.jpg"))
    paths4 = sorted(glob(path + "/202007201035_60m_45도_2_image/*.jpg"))
    pathl = [paths1, paths2, paths3, paths4]
    pathl += pathl

    yolov10_trt = YOLOv10_TRT("models/{}-b{}-{}.engine".format(args.model, args.batch, args.quantization))

    th_load_images = threading.Thread(target=yolov10_trt.load_images, args=(pathl[:args.batch]))
    th_infer_images = threading.Thread(target=yolov10_trt.run)

    th_load_images.start()
    th_infer_images.start()

    th_load_images.join()
    th_infer_images.join()
