import torch
from torch import nn
from torch import nn
from torchsummary import summary

class ModifiedLeNet5(nn.Module):
    """ Modified LeNet network (the code was originally copied and then modified from https://gist.github.com/erykml/cf4e23cf3ab8897b287754dcb11e2f84#file-lenet_network-py)

    Args:
        n_classes (int): number of output classes
    Returns:
        nn.Module: the LeNet network
    """

    def __init__(self, n_classes):
        super(ModifiedLeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.Softmax(logits)
        return logits, probs

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
    model = ModifiedLeNet5(10)
    print(model)
    summary(model,(1,28,28))