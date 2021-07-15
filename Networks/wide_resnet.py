import torch
import torchvision
from torch import nn
from torch import nn
from torchsummary import summary

class WideResNet(nn.Module):
    def __init__(self,pretrained=False,num_classes=10):
        super(WideResNet, self).__init__()
        self.wide_resnet = torchvision.models.wide_resnet50_2(pretrained)
        self.wide_resnet.fc = nn.Linear(2048,num_classes)

    def forward(self,x):
        if x.shape[-3] == 1:
            x = torch.cat([x]*3, dim=-3)
        logit = self.wide_resnet(x)
        probs = nn.Softmax(logit)
        return logit, probs
    def loss(self, output_score, target_label):
        criterion = nn.CrossEntropyLoss()
        return criterion(output_score, target_label)

    def train_step(self, image, target_label, opt):
        output_logits, _ = self.forward(image)
        obj = self.loss(output_logits, target_label)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item()  # report the loss as a float

    def evaluate_accuracy(self, image, target_label):
        # compute the logits
        output_logit, _ = self.forward(image)

        # get the predictions
        _, predicted = torch.max(output_logit, dim=1)

        # get the total number of labels
        total = target_label.size(0)

        # compute the accuracy
        accuracy = ((predicted == target_label).sum()).item() / total

        return accuracy, total
        
if __name__ == "__main__":
    pretrained = False
    wide_resnet = WideResNet(pretrained)
    summary(wide_resnet,(1,28,28))
