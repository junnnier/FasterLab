from models.vgg11 import VGG, make_layers, cfg


def create_model(param):
    class_num = param["category"]
    activations = param["activations"]
    return VGG(make_layers(cfg['B'], batch_norm=True), class_num, activation=activations)


if __name__ == '__main__':
    from torchsummary import summary
    param = {"category": 2,
             "activations": None}
    net = create_model(param)
    result=summary(net,input_size=(3,299,299))
    print(result)