from torchsummary import summary
from Networks.wide_resnet import WideResNet

def create_wide_resnet(pretrained):
    return WideResNet(pretrained)

if __name__ == "__main__":
    pretrained = False
    model = create_wide_resnet(pretrained)
    summary(model,(1,28,28))