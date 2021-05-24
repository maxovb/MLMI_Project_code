from CNPs.CNP import CNP
from CNPs.ConvCNP import  OnTheGridConvCNP
from torchsummary import summary

def create_model(model_name):
    """ Create and return the appropriate CNP model

    Args:
        model_name (string): one of ["CNP", "ConvCNP", "ConvCNPXL", "UNetCNP", "UNet_restrained"]
    Returns:
        nn.Module: instance of the model
        bool: whether the model is a convolutional model
    """
    #TODO: change model name to UNetCNP_restrained  

    assert model_name in ["CNP", "ConvCNP", "ConvCNPXL", "UNetCNP", "UNetCNP_restrained"], "model name: " + model_name + ", not supported"

    convolutional = False

    if model_name == "CNP":

        # parameters
        encoder_layer_widths = [3, 128, 128, 128]
        decoder_layer_widths = [2, 128, 128, 128, 128, 2]

        # create the model
        model = CNP(encoder_layer_widths, decoder_layer_widths)

    elif model_name == "ConvCNP":

        # parameters
        type_CNN = "CNN"
        num_input_channels, num_output_channels = 1, 2
        num_of_filters = 128
        kernel_size_first_convolution, kernel_size_CNN = 9, 5
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_dense_layers = 5  # 4 hidden layers
        num_units_dense_layer = 64

        # create the model
        model = OnTheGridConvCNP(type_CNN, num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution,
                                 kernel_size_CNN, num_convolutions_per_block, num_dense_layers,num_units_dense_layer,
                                 num_residual_blocks)

        # it is a convolutional model
        convolutional = True

    elif model_name == "ConvCNPXL":
        # parameters
        type_CNN = "CNN"
        num_input_channels, num_output_channels = 1, 2
        num_of_filters = 128
        kernel_size_first_convolution, kernel_size_CNN = 9, 11
        num_residual_blocks = 6
        num_convolutions_per_block = 2
        num_dense_layers = 5  # 4 hidden layers
        num_units_dense_layer = 64

        # create the model
        model = OnTheGridConvCNP(type_CNN, num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution,
                                 kernel_size_CNN, num_convolutions_per_block, num_dense_layers, num_units_dense_layer,
                                 num_residual_blocks)

        # it is a convolutional model
        convolutional = True

    elif model_name == "UNetCNP":
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_first_convolution = 9
        kernel_size_CNN = 5
        num_convolutions_per_block = 1
        num_dense_layers = 5
        num_units_dense_layers = 64
        num_down_blocks = 4
        num_of_filters_top_UNet = 32
        pooling_size = 2

        model = OnTheGridConvCNP(type_CNN, num_input_channels=num_input_channels,
                                 num_output_channels=num_output_channels,
                                 num_of_filters=num_of_filters,
                                 kernel_size_first_convolution=kernel_size_first_convolution,
                                 kernel_size_CNN=kernel_size_CNN, num_convolutions_per_block=num_convolutions_per_block,
                                 num_dense_layers=num_dense_layers, num_units_dense_layer=num_units_dense_layers,
                                 num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                 pooling_size=pooling_size)

        # it is a convolutional model
        convolutional = True

    elif model_name == "UNetCNP_restrained":
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_first_convolution = 9
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_dense_layers = 5
        num_units_dense_layers = 64
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = 64

        model = OnTheGridConvCNP(type_CNN, num_input_channels=num_input_channels,
                                 num_output_channels=num_output_channels,
                                 num_of_filters=num_of_filters,
                                 kernel_size_first_convolution=kernel_size_first_convolution,
                                 kernel_size_CNN=kernel_size_CNN, num_convolutions_per_block=num_convolutions_per_block,
                                 num_dense_layers=num_dense_layers, num_units_dense_layer=num_units_dense_layers,
                                 num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                 pooling_size=pooling_size, max_size=max_size)

        # it is a convolutional model
        convolutional = True

    return model, convolutional

def create_joint_model(model_name, model_size):
    """ Create and return the a joint model for doing both unsupervised and supervised training

    Args:
        model_name (string): one of ["CNP", "ConvCNP", "ConvCNPXL", "UNetCNP", "UNet_restrained"]
        model_size (string): size of the supervised head, one of ["small", "medium", "large"]
    Returns:
        nn.Module: instance of the model
        bool: whether the model is a convolutional model
    """

    model, convolutional = create_model(model_name)


if __name__ == "__main__":
    model_name = "UNetCNP_restrained"
    model, convolutional = create_model(model_name)
    if convolutional:
        summary(model, [(1, 28, 28), (1, 28, 28)])
    else:
        summary(model, [(784, 2), (784, 1)])
