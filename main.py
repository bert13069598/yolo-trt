import argparse
import threading
from glob import glob

from detect.v8_obb import YOLOv8_OBB_TRT
from detect.v9 import YOLOv9_TRT
from detect.v10 import YOLOv10_TRT

parser = argparse.ArgumentParser(description='YOLO TRT')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov9c')
parser.add_argument('-n', '--nc', type=int, help='number of class', default=15)
parser.add_argument('-b', '--batch', type=int, help='batch number', default=1)
parser.add_argument('-q', '--quantization', type=str, help='when export, fp32 fp16 int8', default='fp32')
args = parser.parse_args()

if __name__ == "__main__":
    path = "/home/deepet/Downloads/datasets/자율주행드론 비행 영상/Training/image"
    paths1 = sorted(glob(path + "/202007171220_60m_45도_2_image/*.jpg"))
    paths2 = sorted(glob(path + "/202007171454_60m_45도_2_image/*.jpg"))
    paths3 = sorted(glob(path + "/202007171527_80m_45도_2_image/*.jpg"))
    paths4 = sorted(glob(path + "/202007201035_60m_45도_2_image/*.jpg"))
    pathl = [paths1, paths2, paths3, paths4]
    pathl += pathl

    if args.model == 'yolov8n-obb-car':
        yolo_trt = YOLOv8_OBB_TRT(model=args.model,
                                  batch=args.batch,
                                  quantization=args.quantization,
                                  nc=args.nc)
    elif args.model == 'yolov9c':
        yolo_trt = YOLOv9_TRT(model=args.model,
                              batch=args.batch,
                              quantization=args.quantization)
    elif args.model == 'yolov10n-car':
        yolo_trt = YOLOv10_TRT(model=args.model,
                              batch=args.batch,
                              quantization=args.quantization)

    th_load_images = threading.Thread(target=yolo_trt.load_images, args=(pathl[:args.batch]))
    th_infer_images = threading.Thread(target=yolo_trt.run)

    th_load_images.start()
    th_infer_images.start()

    th_load_images.join()
    th_infer_images.join()
