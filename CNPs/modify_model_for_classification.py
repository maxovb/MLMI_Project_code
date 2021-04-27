from CNPs.CNP import CNP, CNPClassifier
from CNPs.ConvCNP import  OnTheGridConvCNP, ConvCNPClassifier

def modify_model_for_classification(model,convolutional = False, freeze = True, img_height = None,img_width = None):
    """ Modify the given model and return the supervised version

    Args:
        model (nn.Module): original unsupervised CNP model
        convolutional (bool): whether the model is a convolutional model
    Returns:
        nn.Module: instance of the new classification model
    """
    if convolutional:

        # check that img_height and img_width are supplied to the function, to know the number of units in the first dense layer
        assert img_height and img_width, "For the convolutional model, image height and width must be passed as an argument to the modify_model_for_classification function"

        # convolutional parameters
        num_features_conv = [128, 8, 2]
        kernel_size = 3

        # get the flattened size
        output_height, output_width = img_height, img_width
        for i in range(len(num_features_conv) - 1):
            output_height -= 2 * (kernel_size // 2)
            output_width -= 2 * (kernel_size // 2)
            output_height = (output_height) // 2
            output_width = (output_width) // 2
        flatten_size = output_height * output_width * num_features_conv[-1]

        # dense parameters
        dense_layer_widths = [flatten_size, 64, 64, 10]

        # create the model
        classification_model = ConvCNPClassifier(model, num_features_conv, kernel_size, dense_layer_widths)

        # freeze the weights from the original CNP
        for param in classification_model.encoder.parameters():
            param.requires_grad = False
        for param in classification_model.CNN.parameters():
            param.requires_grad = False

    else:
        # Classification CNP
        classification_head_layer_widths = [128, 64, 64, 10]
        classification_model = CNPClassifier(model, classification_head_layer_widths)

        # freeze the weights from the original CNP
        if freeze:
            for param in classification_model.encoder.parameters():
                param.requires_grad = False

    return classification_model
