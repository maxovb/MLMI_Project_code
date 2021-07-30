import torch
from torch import nn
from torch import nn
from torchsummary import summary

class ModifiedLeNet5(nn.Module):
    """ Modified LeNet network (the code was originally copied and then modified from https://gist.github.com/erykml/cf4e23cf3ab8897b287754dcb11e2f84#file-lenet_network-py)

    Args:
        n_classes (int): number of output classes
        conv_features (list of int): number of features in the convolutional layer
        dense_layer_features (list of int): number of units in the dense layers
        dropout (bool, optional): whether to use dropout or not
    Returns:
        nn.Module: the LeNet network
    """

    def __init__(self, conv_features, dense_layer_widths, dropout=False):
        super(ModifiedLeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=conv_features[0], out_channels=conv_features[1], kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=conv_features[1], out_channels=conv_features[2], kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=conv_features[2], out_channels=conv_features[3], kernel_size=4, stride=1),
            nn.ReLU(),
        )

        if dropout:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=dense_layer_widths[0], out_features=dense_layer_widths[1]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(in_features=dense_layer_widths[1], out_features=dense_layer_widths[2]),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=dense_layer_widths[0], out_features=dense_layer_widths[1]),
                nn.ReLU(),
                nn.Linear(in_features=dense_layer_widths[1], out_features=dense_layer_widths[2]),
            )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.mean(x,dim=[-2,-1])
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
    conv_features = [1, 6, 16, 120]
    dense_layer_widths = [120, 84, 10]
    model = ModifiedLeNet5(conv_features,dense_layer_widths)
    print(model)
    summary(model,(1,28,28))