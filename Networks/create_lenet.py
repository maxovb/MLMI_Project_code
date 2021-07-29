from torchsummary import summary
from Networks.lenet import  ModifiedLeNet5

def create_lenet(model_size,dropout=False, num_channels=1, num_classes=10):
    assert model_size in ["small","medium","large"], "model_size should be one of small, medium or large but " + str(model_size) + " was given"

    if model_size == "small":
        conv_features = [1 * num_channels, 4, 4, 10]
        dense_layer_widths = [10, 10, num_classes]

    elif model_size == "medium":
        conv_features = [1 * num_channels, 4, 8, 64]
        dense_layer_widths = [64, 64, num_classes]

    elif model_size == "large":
        conv_features = [1 * num_channels, 4, 24, 128]
        dense_layer_widths = [128, 128, num_classes]

    return ModifiedLeNet5(conv_features,dense_layer_widths,dropout)

if __name__ == "__main__":
    model_size = "large"
    model = create_lenet(model_size)
    summary(model,(1,28,28))