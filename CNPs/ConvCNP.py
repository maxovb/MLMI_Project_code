import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from Utils.helper_loss import gaussian_logpdf

class OnTheGridConvCNP(nn.Module):
    """On-the-grid version of the Convolutional Conditional Neural Process
        See https://arxiv.org/abs/1910.13556 for details.

        Args:
            num_input_channels (int): number of input channels, i.e. 1 for BW and 3 for RGB
            num_output_channels (int): number of output channels, i.e. 2 (mean+std) for BW and 3 for RGB
            num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
            kernel_size_first_convolution (int): size of the kernel for the first normalized convolution
            kernel_size_CNN (int): size of the kernel for the CNN part
            num_residual_blocks (int): number of residual blocks in the CNN
            num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
            num_dense_layers (int): number of dense layers at the end
            num_units_dense_layer (int): number of nodes in the hidden dense layers
    """
    def __init__(self, num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution, kernel_size_CNN, num_residual_blocks, num_convolutions_per_block, num_dense_layers, num_units_dense_layer):
        super(OnTheGridConvCNP, self).__init__()
        self.encoder = OnTheGridConvCNPEncoder(num_input_channels,num_of_filters,kernel_size_first_convolution)
        self.CNN = OnTheGridConvCNPCNN(num_of_filters,kernel_size_CNN,num_residual_blocks,num_convolutions_per_block)
        self.decoder = OnTheGridConvCNPDecoder(num_of_filters,num_dense_layers, num_units_dense_layer,num_output_channels)

    def forward(self,mask,context_image):
        """Forward pass through the on-the-grid ConvCNP

        Args:
            mask (tensor): binary tensor indicating context pixels with a 1 (batch,img_height,img_width,1)
            context_image (tensor): masked image with 0 everywhere except at context pixels (batch, img_height, img_width, num_input_channels)
        Returns:
            tensor: predicted mean for all pixels (batch, img_height, img_width, num_output_channels)
            tensor: predicted standard deviation for all pixels (batch, img_height, img_width, num_output_channels)
        """
        encoder_output = self.encoder(mask,context_image)
        x = self.CNN(encoder_output)
        mean, std = self.decoder(x)
        return mean, std

    def loss(self,mean,std,target):
        obj = -gaussian_logpdf(target, mean, std, 'batched_mean')
        return obj

    def train_step(self,mask,context_image,target,opt):
        mean, std = self.forward(mask,context_image)
        obj = self.loss(mean,std,target)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item() # report the loss as a float

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])

class OnTheGridConvCNPEncoder(nn.Module):
    """Encoder for the on-the-grid version of the Convolutional Conditional Neural Process.

    Args:
        num_input_channels (int): number of input channels, i.e. 1 for BW and 3 for RGB
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        kernel_size_first_convolution (int): size of the kernel for the first normalized convolution
    """
    def __init__(self,num_input_channels,num_of_filters,kernel_size_first_convolution):
        super(OnTheGridConvCNPEncoder, self).__init__()
        self.num_input_channels = num_input_channels
        self.depthwise_sep_conv = DepthwiseSeparableConv2D(num_input_channels,num_of_filters,kernel_size_first_convolution,enforce_positivity=True)

    def forward(self,mask,context_image):
        """Forward pass through the on-the-grid ConvCNP

        Args:
            mask (tensor): binary tensor indicating context pixels with a 1 (batch,img_height,img_width,1)
            context_image (tensor): masked image with 0 everywhere except at context pixels (batch, img_height, img_width, num_input_channels)
        Returns:
            tensor (int): latent representation of the input context (batch, img_width, img_size, num_of_filters)
        """

        # repeat the mask on the last dimension to have the same shape as the input image
        mask = torch.cat(self.num_input_channels * [mask],dim=1)

        # apply the convolutions
        density = self.depthwise_sep_conv(mask)
        numerator = self.depthwise_sep_conv(context_image)

        # divide the signal by the density
        signal = numerator / torch.clamp(density, min=1e-3)

        return torch.cat([density,signal],dim=1)


class OnTheGridConvCNPCNN(nn.Module):
    """CNN for the on-the-grid version of the Convolutional Conditional Neural Process. See https://arxiv.org/abs/1910.13556 for details.

    Args:
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        kernel_size_CNN (int): size of the kernel for the CNN part
        num_residual_blocks (int): number of residual blocks in the CNN
        num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
    """

    def __init__(self, num_of_filters, kernel_size_CNN, num_residual_blocks, num_convolutions_per_block):
        super(OnTheGridConvCNPCNN, self).__init__()

        # store the layers as a list
        h = nn.ModuleList([])
        for i in range(0, num_residual_blocks):
            if i == 0:  # do not use residual blocks for the first block because the number of channel changes
                h.append(ConvBlock(2 * num_of_filters, num_of_filters, kernel_size_CNN, num_convolutions_per_block,
                                    is_residual=False))
            else:
                h.append(ConvBlock(num_of_filters, num_of_filters, kernel_size_CNN, num_convolutions_per_block,
                                    is_residual=True))
        self.CNN = nn.Sequential(*h)

    def forward(self,input):
        """Forward pass through the CNN for the on-the-grid CNN

        Args:
            input (tensor): latent representation of the input context (batch, img_width, img_size, num_of_filters)
        Returns:
            tensor: output map of the CNN
        """
        return self.CNN(input)



class OnTheGridConvCNPDecoder(nn.Module):
    """Decoder for the on-the-grid version of the Convolutional Conditional Neural Process. See https://arxiv.org/abs/1910.13556 for details.

    Args:
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        num_dense_layers (int): number of dense layers at the end
        num_units_dense_layer (int): number of nodes in the hidden dense layers
        num_output_channels (int): number of output channels, i.e. 2 (mean+std) for BW and 3 for RGB
    """

    def __init__(self,num_of_filters,num_dense_layers, num_units_dense_layer,num_output_channels):
        super(OnTheGridConvCNPDecoder, self).__init__()

        self.num_output_channels = num_output_channels

        # store the layers as a list
        h = nn.ModuleList([])
        for i in range(0, num_dense_layers):
            if i == 0:
                h.append(nn.Conv2d(num_of_filters, num_units_dense_layer, kernel_size=1))
                h.append(nn.ReLU())
            elif i != num_dense_layers - 1:
                h.append(nn.Conv2d(num_units_dense_layer, num_units_dense_layer, kernel_size=1))
                h.append(nn.ReLU())
            else:
                h.append(nn.Conv2d(num_units_dense_layer, num_output_channels, kernel_size=1))
        self.dense_network = nn.Sequential(*h)

    def forward(self,input):
        """Forward pass through the decoder

        Args:
            tensor (int): output map of the CNN (batch, img_width, img_size, num_of_filters)
        Returns:
            tensor: predicted mean for all pixels (batch, img_height, img_width, num_output_channels)
            tensor: predicted standard deviation for all pixels (batch, img_height, img_width, num_output_channels)
        """
        x = self.dense_network(input)
        x = x.permute(0,2,3,1)
        mean, std = torch.split(x, self.num_output_channels // 2, dim=-1)
        std = 0.01 + 0.99 * nn.functional.softplus(std)

        return mean, std


class ConvCNPClassifier(nn.Module):
    """ Modify a CNP to replace the decoder with a classification head
    Args:
        model (nn.module): original CNP
        num_features_conv (list of int): list with the number of filters for each convolution (first entry is the number of channels in the CNN)
        dense_layer_widths (list of int): list with the dimensionality of the layers (first entry is the number of nodes after flattening the output of the last convolution)

    """
    def __init__(self,model, num_features_conv, kernel_size, dense_layer_widths):
        super(ConvCNPClassifier,self).__init__()
        self.encoder = model.encoder
        self.CNN = model.CNN

        # add the convolutions
        l1 = len(num_features_conv)
        h_conv = nn.ModuleList([])  # store the layers as a list
        for i in range(0, l1 - 1):
            h_conv.append(DepthwiseSeparableConv2D(num_features_conv[i],num_features_conv[i+1],kernel_size=kernel_size,padding=False))
            h_conv.append(nn.ReLU())
            h_conv.append(nn.MaxPool2d(kernel_size=2))
        self.conv_network = nn.Sequential(*h_conv)

        # add the dense layers
        l2 = len(dense_layer_widths)
        h_dense = nn.ModuleList([])  # store the layers as a list
        for i in range(0, l2 - 1):
            h_dense.append(nn.Linear(dense_layer_widths[i],dense_layer_widths[i+1]))
            if i != l2 - 2:  # no ReLU for the last layer
                h_dense.append(nn.ReLU())
        self.dense_network = nn.Sequential(*h_dense)
        self.final_activation = nn.Softmax(dim=-1)

    def forward(self,mask,context_img):
        """ Forward pass through the Classification CNP

        Args:
            mask (tensor): binary mask locating context pixels (batch,img_height,img_width,1)
            context_img (tensor): context pixels with non-context points masked (batch,img_height,img_width,num_channels)

        Returns:
            tensor: classification score for the different output classes (batch,num_classes)
            tensor: probability mass for the different output classes (batch,num_classes)
        """
        x = self.encoder(mask,context_img)
        x = self.CNN(x)
        x = self.conv_network(x)
        x = x.permute(0,2,3,1)
        x = torch.flatten(x, start_dim=1)
        output_score = self.dense_network(x)
        output_logit = self.final_activation(output_score)
        return output_score, output_logit

    def loss(self,output_score,target_label):
        criterion = nn.CrossEntropyLoss()
        return criterion(output_score,target_label)

    def train_step(self,mask,context_img,target_label,opt):
        output_score, _ = self.forward(mask,context_img)
        obj = self.loss(output_score,target_label)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item() # report the loss as a float

    def evaluate_accuracy(self,mask,context_img, target_label):
        # compute the logits
        output_score, output_logit = self.forward(mask,context_img)

        # get the predictions
        _, predicted = torch.max(output_logit,dim=1)

        # get the total number of labels
        total = target_label.size(0)

        # compute the accuracy
        accuracy = ((predicted == target_label).sum()).item()/total

        return accuracy, total

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])

class ConvBlock(nn.Module):
    """ Convolutional (optionally residual) block for the one-the-grid ConvCNP

    Args:
        num_of_input_channels (int): number of channels at of the input of the block
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        kernel_size (int): size of the kernel
        num_convolutions_per_block (int): number of convolutional layers per residual blocks
        is_residual (bool): whether it is a residual convolutional block or not
    """

    def __init__(self,num_of_input_channels,num_of_filters,kernel_size,num_convolutions_per_block, is_residual):
        super(ConvBlock, self).__init__()
        h = nn.ModuleList([])
        for i in range(num_convolutions_per_block):
            if i == 0:
                h.append(DepthwiseSeparableConv2D(num_of_input_channels,num_of_filters,kernel_size))
            else:
                h.append(DepthwiseSeparableConv2D(num_of_filters, num_of_filters, kernel_size))
            if i != num_convolutions_per_block-1:
                h.append(nn.ReLU())
        self.block = nn.Sequential(*h)
        self.activation = nn.ReLU()
        self.is_residual = is_residual

    def forward(self,input):
        """Forward pass through the residual block

            Args:
                input (tensor): input tensor (batch,channels,img_height,img_width)
            Returns:
                tensor: output of the residual block
        """
        x = self.block(input)
        if self.is_residual:
            x = x + input
        x = self.activation(x)
        return x

class DepthwiseSeparableConv2D(nn.Module):
    """ Depthwise separable 2D convolution

    Args:
        num_input_channels (int): size of the channel channel dimension of the input
        num_of_filters (int): size of the channel channel dimension of the output
        kernel_size (int): size of the kernel
    """
    def __init__(self,num_input_channels,num_of_filters,kernel_size,enforce_positivity=False,padding=True):
        super(DepthwiseSeparableConv2D,self).__init__()
        if padding:
             padding = kernel_size//2
        else:
            padding = 0

        if enforce_positivity:
            self.depthwise = make_abs_conv(nn.Conv2d)(num_input_channels, num_input_channels, kernel_size=kernel_size,padding=padding,groups=num_input_channels)
            self.pointwise = make_abs_conv(nn.Conv2d)(num_input_channels, num_of_filters, kernel_size=1)
        else:
            self.depthwise = nn.Conv2d(num_input_channels, num_input_channels, kernel_size=kernel_size,padding=padding,groups=num_input_channels)
            self.pointwise = nn.Conv2d(num_input_channels, num_of_filters, kernel_size=1)

    def forward(self,input):
        """ Forward pass through the depthwise separable 2D convolution

            Args:
                input (tensor): input tensor (batch,channels,img_height,img_width)
            Returns:
                tensor: output of the depthwise separable 2D convolution
        """
        x = self.depthwise(input)
        x = self.pointwise(x)
        return x

def make_abs_conv(Conv):
    """Make a convolution have only positive parameters. (copied from https://github.com/YannDubs/Neural-Process-Family.git)"""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(
                input,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv

if __name__ == "__main__":
    img_height = 28
    img_width = 28

    model = OnTheGridConvCNP(num_input_channels=1,num_output_channels=2,num_of_filters=128,kernel_size_first_convolution=9,kernel_size_CNN=5,num_residual_blocks=4,num_convolutions_per_block=1,num_dense_layers=5, num_units_dense_layer=64)
    summary(model, [(1, img_height, img_width), (1, img_height, img_width)])

    # Classfication CNP
    num_features_conv = [128, 8, 2]
    kernel_size = 3
    # get the flattened size
    output_height,output_width = img_height,img_width
    for i in range(len(num_features_conv)-1):
        output_height -= 2 * (kernel_size//2)
        output_width -= 2 * (kernel_size // 2)
        output_height = (output_height)//2
        output_width = (output_width)//2

    flatten_size = output_height * output_width * num_features_conv[-1]
    dense_layer_widths = [flatten_size,64,64,10]
    classification_model = ConvCNPClassifier(model, num_features_conv, kernel_size, dense_layer_widths)

    # freeze the weights from the original CNP
    for param in classification_model.encoder.parameters():
        param.requires_grad = False
    for param in classification_model.CNN.parameters():
        param.requires_grad = False

    summary(classification_model, [(1, img_height, img_width), (1, img_height, img_width)])









