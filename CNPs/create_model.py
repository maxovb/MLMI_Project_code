from CNPs.CNP import CNP
from CNPs.ConvCNP import  OnTheGridConvCNP

def create_model(model_name):
    """ Create and return the appropriate CNP model

    Args:
        model_name (string): one of ["CNP", "ConvCNP", "ConvCNPXL"]
    Returns:
        nn.Module: instance of the model
        bool: whether the model is a convolutional model
    """

    assert model_name in ["CNP", "ConvCNP", "ConvCNPXL"], "model name: " + model_name + ", not supported"

    convolutional = False

    if model_name == "CNP":

        # parameters
        encoder_layer_widths = [3, 128, 128, 128]
        decoder_layer_widths = [2, 128, 128, 128, 128, 2]

        # create the model
        model = CNP(encoder_layer_widths, decoder_layer_widths)

    elif model_name == "ConvCNP":

        # parameters
        num_input_channels, num_output_channels = 1, 2
        num_of_filters = 128
        kernel_size_first_convolution, kernel_size_CNN = 9, 5
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_dense_layers = 5  # 4 hidden layers
        num_units_dense_layer = 64

        # create the model
        model = OnTheGridConvCNP(num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution,
                               kernel_size_CNN, num_residual_blocks, num_convolutions_per_block, num_dense_layers,
                               num_units_dense_layer)

        # it is a convolutional model
        convolutional = True

    elif model_name == "ConvCNPXL":
        # parameters
        num_input_channels, num_output_channels = 1, 2
        num_of_filters = 128
        kernel_size_first_convolution, kernel_size_CNN = 9, 11
        num_residual_blocks = 6
        num_convolutions_per_block = 2
        num_dense_layers = 5  # 4 hidden layers
        num_units_dense_layer = 64

        # create the model
        model = OnTheGridConvCNP(num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution,
                               kernel_size_CNN, num_residual_blocks, num_convolutions_per_block, num_dense_layers,
                               num_units_dense_layer)

        # it is a convolutional model
        convolutional = True

    return model, convolutional