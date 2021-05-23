from CNPs.CNP import CNP, CNPClassifier
from CNPs.ConvCNP import  OnTheGridConvCNP, ConvCNPClassifier, ConvCNPExtractRepresentation
import torch

def modify_model_for_classification(model,model_size,convolutional = False, freeze = True, img_height = None, img_width = None, num_channels = None, layer_id = None, pooling = None, joint = False):
    """ Modify the given model and return the supervised version

    Args:
        model (nn.Module): original unsupervised CNP model
        model_size (string): one of "small", "medium" and "large, influences the number of parameters in the classification head
        convolutional (bool): whether the model is a convolutional model
        freeze (bool): whether to freeze the weights from the original CNP
        img_height (int): height of the image in pixels, to pass when convolutional is True
        img_width (int): width of the image in pixels, to pass when convolutional is True
        num_channels (int): number of channels of the input image, to pass when convolutional is True
        layer_id (int): layer from which to extract the latent representation, to pass when convolutional is True
        pooling (string): type of pooling to apply on the representation, to pass when convolutional is True
        joint (bool): whether the model will be used for joint training or not
    Returns:
        nn.Module: instance of the new classification model
    """
    assert model_size in ["small","medium","large"], "model_size should be one of 'small','medium' or 'large', not " + str(model_size)

    if convolutional:

        # check that img_height and img_width are supplied to the function, to know the number of units in the first dense layer
        assert img_height and img_width and num_channels , "For the convolutional model, image height, image width and" \
                                                           " num_channels  must be passed as an argument to the" \
                                                           " modify_model_for_classification function"

        assert layer_id and pooling , "For the convolutional model, layer_id and pooling type must be passed as an" \
                                      " argument to the modify_model_for_classification function"
        # parameters
        temp_x = torch.randn(2,num_channels,img_height,img_width)
        temp_model = ConvCNPExtractRepresentation(model,layer_id,pooling)
        out = temp_model(temp_x,temp_x)
        print(out.shape)
        tmp, r_size = out.shape
        if model_size == "small":
            dense_layer_widths = [r_size,10,10]
        elif model_size == "medium":
            dense_layer_widths = [r_size,64,64,10]
        elif model_size == "large":
            dense_layer_widths = [r_size, 128, 128, 128, 128, 10]

        # create the model
        classification_model = ConvCNPClassifier(model, dense_layer_widths, layer_id=layer_id, pooling=pooling)

        # freeze the weights from the original CNP
        if freeze:
            for param in classification_model.encoder.parameters():
                param.requires_grad = False
            for param in classification_model.CNN.parameters():
                param.requires_grad = False

    else:

        if model_size == "small":
            classification_head_layer_widths = [128,10,10]
        elif model_size == "medium":
            classification_head_layer_widths = [128, 64, 64, 10]
        elif model_size == "large":
            classification_head_layer_widths = [128, 128, 128, 128, 128, 10]

        # create the model
        classification_model = CNPClassifier(model, classification_head_layer_widths)

        # freeze the weights from the original CNP
        if freeze:
            for param in classification_model.encoder.parameters():
                param.requires_grad = False

    return classification_model
