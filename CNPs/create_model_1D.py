from CNPs.CNP import CNP
from CNPs.NP import NP
#from CNPs.ConvCNP import  OnTheGridConvCNP
from CNPs.OffTheGridConvCNP import OffTheGridConvCNP
from torchsummary import summary

def create_model_off_the_grid(model_name, model_size=None, num_classes=10):
    """ Create and return the appropriate CNP model

    Args:
        model_name (string): one of ["CNP", "ConvCNP", "ConvCNPXL", "UNetCNP", "UNet_restrained", "NP_UG", "UNetCNP_GMM", "UNetCNP_restrained_GMM"]
        model_size (string), optional: size of the classification part of the network for the GMM models
        num_classes (int), optional: number of classes
    Returns:
        nn.Module: instance of the model
        bool: whether the model is a convolutional model
    """

    assert model_name in ["CNP", "ConvCNP", "ConvCNPXL", "UNetCNP", "UNetCNP_restrained", "NP_UG", "NP_UG_DT",
                          "UNetCNP_GMM","UNetCNP_restrained_GMM","UNetCNP_GMM_blocked","UNetCNP_restrained_GMM_blocked"]\
                         , "model name: " + model_name + ", not supported"

    if model_name == "CNP":

        # parameters
        encoder_layer_widths = [2, 128, 128, 128]
        decoder_layer_widths = [1, 128, 128, 128, 128, 2]

        # create the model
        model = CNP(encoder_layer_widths, decoder_layer_widths)

    elif model_name == "NP_UG":
        encoder_layer_widths = [2, 128, 128]
        decoder_layer_widths = [1, 128, 128, 128, 1]
        classifier_layer_widths = [128, 128, num_classes]
        latent_network_layer_widths = [128 + num_classes, 128, 128]
        prior = "UnitGaussian"
        model = NP(encoder_layer_widths, decoder_layer_widths, classifier_layer_widths, latent_network_layer_widths,
                   prior)

    elif model_name == "NP_UG_DT":
        encoder_layer_widths = [2, 128, 128]
        decoder_layer_widths = [1, 128, 128, 128, 1]
        classifier_layer_widths = [128, 128, num_classes]
        latent_network_layer_widths = [128 + num_classes, 128, 128]
        prior = "UnitGaussian"
        deterministic = True
        model = NP(encoder_layer_widths, decoder_layer_widths, classifier_layer_widths, latent_network_layer_widths,
                   prior, deterministic=deterministic)

    elif model_name == "ConvCNP":

        # parameters
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "CNN"
        num_input_channels, num_output_channels = 1, 2
        num_of_filters = 128
        kernel_size_first_convolution, kernel_size_CNN = 9, 5
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_dense_layers = 5  # 4 hidden layers
        num_units_dense_layer = 64

        # create the model
        model = OffTheGridConvCNP(learn_length_scale, points_per_unit, type_CNN, num_input_channels,
                                  num_output_channels, num_of_filters, kernel_size_first_convolution, kernel_size_CNN,
                                  num_convolutions_per_block, num_dense_layers, num_units_dense_layer,
                                  num_residual_blocks)

    elif model_name == "ConvCNPXL":
        # parameters
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "CNN"
        num_input_channels, num_output_channels = 1, 2
        num_of_filters = 128
        kernel_size_first_convolution, kernel_size_CNN = 9, 11
        num_residual_blocks = 6
        num_convolutions_per_block = 2
        num_dense_layers = 5  # 4 hidden layers
        num_units_dense_layer = 64

        # create the model
        model = OffTheGridConvCNP(learn_length_scale,points_per_unit,type_CNN, num_input_channels, num_output_channels,
                                 num_of_filters, kernel_size_first_convolution, kernel_size_CNN,
                                 num_convolutions_per_block, num_dense_layers, num_units_dense_layer,
                                 num_residual_blocks)

    elif model_name == "UNetCNP":
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = None

        if model_size == "LR":
            classifier_layer_widths = [64, num_classes]
        elif model_size == "small":
            classifier_layer_widths = [64, 10, num_classes]
        elif model_size == "medium":
            classifier_layer_widths = [64, 64, 64, num_classes]
        elif model_size == "large":
            classifier_layer_widths = [64, 128, 128, 128, 128, num_classes]

        model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                                  type_CNN=type_CNN, num_input_channels=num_input_channels,
                                  num_output_channels=num_output_channels,
                                  num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                                  num_convolutions_per_block=num_convolutions_per_block,
                                  num_residual_blocks=num_residual_blocks,
                                  num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                  pooling_size=pooling_size, max_size=max_size)

    elif model_name == "UNetCNP_restrained":
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = 64

        if model_size == "LR":
            classifier_layer_widths = [64, num_classes]
        elif model_size == "small":
            classifier_layer_widths = [64, 10, num_classes]
        elif model_size == "medium":
            classifier_layer_widths = [64, 64, 64, num_classes]
        elif model_size == "large":
            classifier_layer_widths = [64, 128, 128, 128, 128, num_classes]

        model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                                  type_CNN=type_CNN, num_input_channels=num_input_channels,
                                  num_output_channels=num_output_channels,
                                  num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                                  num_convolutions_per_block=num_convolutions_per_block,
                                  num_residual_blocks=num_residual_blocks,
                                  num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                  pooling_size=pooling_size, max_size=max_size)

    elif model_name == "UNetCNP_GMM":
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = None
        is_gmm = True
        num_classes = num_classes

        if model_size == "LR":
            classifier_layer_widths = [64, num_classes]
        elif model_size == "small":
            classifier_layer_widths = [64, 10, num_classes]
        elif model_size == "medium":
            classifier_layer_widths = [64, 64, 64, num_classes]
        elif model_size == "large":
            classifier_layer_widths = [64, 128, 128, 128, 128, num_classes]

        model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                                  type_CNN=type_CNN, num_input_channels=num_input_channels,
                                  num_output_channels=num_output_channels,
                                  num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                                  num_convolutions_per_block=num_convolutions_per_block,
                                  num_residual_blocks=num_residual_blocks,
                                  num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                  pooling_size=pooling_size, max_size=max_size, is_gmm=is_gmm,
                                  classifier_layer_widths=classifier_layer_widths, num_classes=num_classes)

    elif model_name == "UNetCNP_restrained_GMM":
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = 64
        is_gmm = True
        num_classes = num_classes

        if model_size == "LR":
            classifier_layer_widths = [64, num_classes]
        elif model_size == "small":
            classifier_layer_widths = [64, 10, num_classes]
        elif model_size == "medium":
            classifier_layer_widths = [64, 64, 64, num_classes]
        elif model_size == "large":
            classifier_layer_widths = [64, 128, 128, 128, 128, num_classes]

        model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                                  type_CNN=type_CNN, num_input_channels=num_input_channels,
                                  num_output_channels=num_output_channels,
                                  num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                                  num_convolutions_per_block=num_convolutions_per_block,
                                  num_residual_blocks=num_residual_blocks,
                                  num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                  pooling_size=pooling_size, max_size=max_size, is_gmm=is_gmm,
                                  classifier_layer_widths=classifier_layer_widths, num_classes=num_classes)

    elif model_name == "UNetCNP_GMM_blocked":
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = None
        is_gmm = True
        num_classes = num_classes
        block_center_connections = True

        if model_size == "LR":
            classifier_layer_widths = [64, num_classes]
        elif model_size == "small":
            classifier_layer_widths = [64, 10, num_classes]
        elif model_size == "medium":
            classifier_layer_widths = [64, 64, 64, num_classes]
        elif model_size == "large":
            classifier_layer_widths = [64, 128, 128, 128, 128, num_classes]

        model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                                  type_CNN=type_CNN, num_input_channels=num_input_channels,
                                  num_output_channels=num_output_channels,
                                  num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                                  num_convolutions_per_block=num_convolutions_per_block,
                                  num_residual_blocks=num_residual_blocks,
                                  num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                  pooling_size=pooling_size, max_size=max_size, is_gmm=is_gmm,
                                  classifier_layer_widths=classifier_layer_widths, num_classes=num_classes,
                                  block_center_connection = block_center_connections)

    elif model_name == "UNetCNP_restrained_GMM_blocked":
        learn_length_scale = True
        points_per_unit = 10
        type_CNN = "UNet"
        num_input_channels = 1
        num_output_channels = 2
        num_of_filters = 128
        kernel_size_CNN = 3
        num_residual_blocks = 4
        num_convolutions_per_block = 1
        num_down_blocks = 4
        num_of_filters_top_UNet = 64
        pooling_size = 2
        max_size = 64
        is_gmm = True
        num_classes = num_classes
        block_center_connections = True

        if model_size == "LR":
            classifier_layer_widths = [64, num_classes]
        elif model_size == "small":
            classifier_layer_widths = [64, 10, num_classes]
        elif model_size == "medium":
            classifier_layer_widths = [64, 64, 64, num_classes]
        elif model_size == "large":
            classifier_layer_widths = [64, 128, 128, 128, 128, num_classes]

        model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                                  type_CNN=type_CNN, num_input_channels=num_input_channels,
                                  num_output_channels=num_output_channels,
                                  num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                                  num_convolutions_per_block=num_convolutions_per_block,
                                  num_residual_blocks=num_residual_blocks,
                                  num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                                  pooling_size=pooling_size, max_size=max_size, is_gmm=is_gmm,
                                  classifier_layer_widths=classifier_layer_widths, num_classes=num_classes,
                                  block_center_connection=block_center_connections)

    return model

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
