import argparse
import torch
from utils.select_tools import select_device


def pytorch_to_onnx(model, image, file, opset, train, dynamic, dynamic_output, simplify):
    import onnx
    f = str(file).replace('.pth', '.onnx')
    try:
        torch.onnx.export(model, image, f,
                          verbose=False,
                          opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},
                                        'output': dynamic_output
                                        } if dynamic else None)
        model_onnx = onnx.load(f)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim
            try:
                print("simplifying with onnx-simplifier {}".format(onnxsim.__version__))
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(image.shape)} if dynamic else None)
                assert check, 'simplify check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print("simplifier failure: {}".format(e))

        print("export success, saved as: {}".format(f))

    except Exception as e:
        print("export failure: {}".format(e))


def load_model(model, map_location):
    ckpt = torch.load(model, map_location=map_location)
    return ckpt["model"].float().eval()


def main(opt):
    device = select_device(str(opt.device))
    assert not (device == "cpu" and opt.half), "--half only compatible with GPU export, i.e. use --device 0"
    # 加载模型
    model = load_model(opt.weight, device)  # load FP32 model
    image = torch.zeros(opt.batch_size, 3, *opt.imgsz).to(device)
    if opt.half:
        image, model = image.half(), model.half()
    model.eval()
    # 测试运行
    y = model(image)
    print("input {} in {}\noutput {} in {}".format(image.shape, image.device, y.shape, y.device))
    # 转换模型
    if opt.mode == "onnx":
        pytorch_to_onnx(model, image, opt.weight, opt.opset, opt.train, opt.dynamic, opt.dynamic_output, opt.simplify)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default="", help='Model weight path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[224, 224], help='Height and width of the image, default=[224, 224]')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size. default=1')
    parser.add_argument('--device', type=str, default="0", help="cuda device, 0 or 0,1,2,3 or cpu. default=0")
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--dynamic-output', type=dict, default={0: 'batch', 2: 'height', 3: 'width'},
                        help='ONNX: Parameter content of "output" in dynamic axes. default={0:"batch", 2:"height", 3:"width"}')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=13, help='ONNX: opset version. default=13')
    parser.add_argument('--mode', default="onnx", choices=['onnx'], help='available formats are (torchscript, onnx), default=onnx')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)