from CNPs.CNP import CNP, CNPClassifier
from CNPs.ConvCNP import  OnTheGridConvCNP, ConvCNPClassifier

def modify_model_for_classification(model,model_size,convolutional = False, freeze = True, img_height = None,img_width = None):
    """ Modify the given model and return the supervised version

    Args:
        model (nn.Module): original unsupervised CNP model
        model_size (string): one of "small", "medium" and "large, indicates the number of parameters in the classification head
        convolutional (bool): whether the model is a convolutional model
    Returns:
        nn.Module: instance of the new classification model
    """
    assert model_size in ["small","medium","large"], "model_size should be one of 'small','medium' or 'large', not " + str(model_size)

    if convolutional:

        # check that img_height and img_width are supplied to the function, to know the number of units in the first dense layer
        assert img_height and img_width, "For the convolutional model, image height and width must be passed as an argument to the modify_model_for_classification function"

        # dense parameters
        if model_size == "small":
            dense_layer_widths = [128,10,10]
        elif model_size == "medium":
            dense_layer_widths = [128,64,64,10]
        elif model_size == "large":
            dense_layer_widths = [128, 128, 128, 128, 128, 10]

        # create the model
        classification_model = ConvCNPClassifier(model, dense_layer_widths)

        # freeze the weights from the original CNP
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
