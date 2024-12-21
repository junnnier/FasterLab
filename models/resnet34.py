from models.resnet18 import ResNet, BasicBlock


def create_model(param):
    """ return a ResNet 34 object"""
    class_num = param["category"]
    activations = param["activations"]
    return ResNet(BasicBlock, [3, 4, 6, 3], class_num, activation=activations)


if __name__ == '__main__':
    from torchsummary import summary
    param = {"category": 2,
             "activations": None}
    net=create_model(param)
    result=summary(net,input_size=(3,299,299))
    print(result)