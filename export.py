import tensorrt as trt
import argparse

parser = argparse.ArgumentParser(description='YOLO export')
parser.add_argument('-m', '--model', type=str, help='model name for .pt', default='yolov8n-obb')
parser.add_argument('-b', '--batch', type=int, help='batch number', default=1)
parser.add_argument('-q', '--quantization', type=str, help='when export, fp32 fp16 int8', default='fp32')
args = parser.parse_args()

file_path = "{}-b{}-{}".format(args.model, args.batch, args.quantization)

# Export the trt model
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
        builder.create_builder_config() as config, \
        trt.OnnxParser(network, TRT_LOGGER) as parser, \
        trt.Runtime(TRT_LOGGER) as runtime:
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 MiB
    print("fp16 :", builder.platform_has_fast_fp16)
    print("int8 :", builder.platform_has_fast_int8)
    print('Loading ONNX file from path {}...'.format(file_path + '.onnx'))

    print('Beginning ONNX file parsing')
    success = parser.parse_from_file(file_path + '.onnx')
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        print('ERROR: Failed to parse the ONNX file.')
    print('Completed parsing of ONNX file')

    print('Building an engine file from {}...'.format(file_path + '.onnx'))
    plan = builder.build_serialized_network(network, config)
    # engine = runtime.deserialize_cuda_engine(plan)
    print("Completed creating Engine")
    with open(file_path + '.engine', "wb") as f:
        f.write(plan)
