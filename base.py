import os
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time, sleep

import cv2
import numpy as np
import tensorrt as trt
import torch


def get_warpAffineM(W, H, dst_width=1024, dst_height=1024):
    scale = min((dst_width / W, dst_height / H))
    ox = (dst_width - scale * W) / 2
    oy = (dst_height - scale * H) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    IM = cv2.invertAffineTransform(M)
    return M, IM


def preprocess_warpAffine(image, M, dst_width=1024, dst_height=1024):
    img_pre = cv2.warpAffine(image, M,
                             (dst_width, dst_height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(114, 114, 114))
    return img_pre


class TRT:
    def __init__(self, trt_engine, batch, quantization, src_shape, dst_shape):
        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda:0')
            print(f'Device: {self.DEVICE}', torch.cuda.get_device_name(self.DEVICE))
        else:
            self.DEVICE = torch.device('cpu')
            print(f'Device: {self.DEVICE}')

        self.batch = batch
        if quantization == 'fp32':
            self.dtype = torch.float32
        elif quantization == 'fp16':
            self.dtype = torch.float16

        self.executor = ThreadPoolExecutor(max_workers=min(self.batch, os.cpu_count()))
        self.q = queue.Queue(maxsize=2)

        self.src_shape = src_shape
        self.dst_shape = dst_shape
        self.M, self.IM = get_warpAffineM(*self.src_shape, *self.dst_shape)

        self.input_buffer = torch.empty((self.batch, 3, self.dst_shape[1], self.dst_shape[0]),
                                        dtype=self.dtype,
                                        device=self.DEVICE)
        self.output_buffer = None
        self.stream = torch.cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(trt_engine, "rb") as f:
            serialized_engine = f.read()
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

    def run(self):
        while True:
            while self.q.empty():
                sleep(0.001)
            t0 = time()
            imgs = self.q.get()
            t1 = time()
            pre_batch = self.preprocess(*imgs)
            t2 = time()
            results = self.infer(pre_batch)
            t3 = time()
            post_batch = self.postprocess(*imgs, results=results)
            t4 = time()

            print('{}\t{:.5f} + {:.5f} + {:.5f} + {:.5f} = {:.5f}'.format(self.q.qsize(),
                                                                          t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0))

            for i in range(self.batch):
                cv2.imshow(f"Drone {i + 1}",
                           cv2.resize(post_batch[i], (self.src_shape[0] // 4, self.src_shape[1] // 4)))
            cv2.waitKey(1)

    def load_images(self, *pathsl):
        imgs = [None] * self.batch
        for paths in zip(*pathsl):
            while self.q.full():
                sleep(0.001)
            futures = {self.executor.submit(lambda x: cv2.imread(x), path): i for i, path in enumerate(paths)}
            for future in as_completed(futures):
                imgs[futures[future]] = future.result()
            self.q.put(imgs)

    def preprocess(self, *imgs):
        img_pre_batch = np.stack([preprocess_warpAffine(img, self.M, *self.dst_shape) for img in imgs])
        img_pre_batch = (img_pre_batch[..., ::-1]).transpose(0, 3, 1, 2)
        img_pre_batch = np.ascontiguousarray(img_pre_batch)
        img_pre_batch = torch.from_numpy(img_pre_batch).to(self.DEVICE).float()
        img_pre_batch /= 255
        return img_pre_batch

    def infer(self, img_pre_batch):
        self.input_buffer.copy_(img_pre_batch)
        with self.engine.create_execution_context() as context:
            context.set_tensor_address("images", self.input_buffer.data_ptr())
            context.set_tensor_address("output0", self.output_buffer.data_ptr())

            context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            self.stream.synchronize()
        return self.output_buffer

    def postprocess(self, *imgs, results, conf_thres=0.25, iou_thres=0.45):
        raise NotImplementedError()
