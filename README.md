# yolo-trt

## yolov8-obb
### export
```bash
python3 export.py -m yolov8n-obb -b 4
```

### run
```bash
python yolov8-obb-trt.py \
-b 4 \
-m yolov8n-car \
-q fp32 \
-n 1
```

## yolov9

### run
```bash
python yolov9-trt.py \
-b 4 \
-m yolov9c \
-q fp32
```

## yolov10
### run
```bash
python yolov10-trt.py \
-m yolov10n-car \
-b 4 \
-q fp32
```
