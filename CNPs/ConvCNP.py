import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from Utils.helper_loss import gaussian_logpdf

class OnTheGridConvCNP(nn.Module):
    """On-the-grid version of the Convolutional Conditional Neural Process
        See https://arxiv.org/abs/1910.13556 for details.

        Args:
            type_CNN (string): one of ["CNN","UNet"]
            num_input_channels (int): number of input channels, i.e. 1 for BW and 3 for RGB
            num_output_channels (int): number of output channels, i.e. 2 (mean+std) for BW and 3 for RGB
            num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
            kernel_size_first_convolution (int): size of the kernel for the first normalized convolution
            kernel_size_CNN (int): size of the kernel for the CNN part
            num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
            num_dense_layers (int): number of dense layers at the end
            num_units_dense_layer (int): number of nodes in the hidden dense layers
            num_residual_blocks (int): number of residual blocks in the CNN
            num_down_blocks (int): number of down blocks when using UNet
            num_of_filters_top_UNet (int): number of filters for the top UNet layer (doubles all the way down)
            pooling_size (int): pooling size for the UNet
            max_size (int or None): maximum number of features in the UNet
    """
    def __init__(self, type_CNN, num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution, kernel_size_CNN, num_convolutions_per_block, num_dense_layers, num_units_dense_layer, num_residual_blocks = None, num_down_blocks=None, num_of_filters_top_UNet=None, pooling_size=None, max_size = None):
        super(OnTheGridConvCNP, self).__init__()

        self.encoder = OnTheGridConvCNPEncoder(num_input_channels,num_of_filters,kernel_size_first_convolution)

        if type_CNN == "CNN":
            assert num_residual_blocks, "The argument num_residual blocks should be passed as integer when using the CNN ConvCNP"
            self.CNN = OnTheGridConvCNPCNN(num_of_filters,kernel_size_CNN,num_residual_blocks,num_convolutions_per_block)

        elif type_CNN == "UNet":
            assert num_down_blocks and num_of_filters_top_UNet and pooling_size, "Arguments num_down_blocks, num_of_filters_top_UNet and pooling_size should be passed as integers when using the UNet ConvCNP"
            self.CNN = OnTheGridConvCNPUNet(num_of_filters_top_UNet, 2 * num_of_filters, kernel_size_CNN, num_down_blocks, num_convolutions_per_block, pooling_size, max_size)

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
        self.h = nn.ModuleList([])
        self.num_residual_blocks = num_residual_blocks
        for i in range(0, num_residual_blocks):
            if i == 0:  # do not use residual blocks for the first block because the number of channel changes
                self.h.append(ConvBlock(2 * num_of_filters, num_of_filters, kernel_size_CNN, num_convolutions_per_block,
                                    is_residual=False))
            else:
                self.h.append(ConvBlock(num_of_filters, num_of_filters, kernel_size_CNN, num_convolutions_per_block,
                                    is_residual=True))

    def forward(self,input, layer_id=-1):
        """Forward pass through the CNN for the on-the-grid CNN

        Args:
            input (tensor): latent representation of the input context (batch, img_width, img_size, num_of_filters)
            layer_id (int), optional: id of the layer to output, by default -1 to return the last one
        Returns:
            tensor: output map of the CNN
        """
        x = input
        layers = []
        for i in range(self.num_residual_blocks):
            x = self.h[i](x)
            layers.append(x)
        return layers[layer_id]

class OnTheGridConvCNPUNet(nn.Module):
    """U-Net for the CNN part of the on-the-grid version of the Convolutional Conditional Neural Process.

        Args:
            num_in_filters (int): number of filters taken in the network
            num_of_filters (int): number of filters per convolution at the top, doubles at every step down, and divides by two at every step up
            kernel_size_CNN (int): size of the kernel
            num_down_blocks (int): number of blocks until the bottleneck
            num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
            pooling_size (int): size of the maxpooling layers
    """
    def __init__(self, num_of_filters, num_in_filters, kernel_size_CNN, num_down_blocks, num_convolutions_per_block, pooling_size, max_size=None):
        super(OnTheGridConvCNPUNet, self).__init__()

        # store some variables
        self.num_down_blocks = num_down_blocks

        self.pool = torch.nn.MaxPool2d(pooling_size)
        self.upsample = torch.nn.Upsample(scale_factor=pooling_size)
        self.h_down = nn.ModuleList([])
        for i in range(num_down_blocks):
            if i == 0:  # do not use residual blocks for the first block because the number of channel changes
                if max_size:
                    num_out = min(num_of_filters,max_size)
                else:
                    num_out = num_of_filters
                self.h_down.append(ConvBlock(num_in_filters, num_out, kernel_size_CNN,
                                             num_convolutions_per_block, is_residual=False))
            else:
                if max_size:
                    num_in = min((2**(i-1)) * num_of_filters,max_size)
                    num_out = min((2**(i)) * num_of_filters,max_size)
                else:
                    num_in = (2 ** (i - 1)) * num_of_filters
                    num_out = (2 ** (i)) * num_of_filters
                self.h_down.append(ConvBlock(num_in, num_out, kernel_size_CNN,num_convolutions_per_block,
                                             is_residual=False))

        if max_size:
            num = min((2**(num_down_blocks-1)) * num_of_filters, max_size)
        else:
            num = (2**(num_down_blocks-1)) * num_of_filters
        self.h_bottom = ConvBlock(num, num, kernel_size_CNN,num_convolutions_per_block, is_residual=False)

        self.h_up = nn.ModuleList([])
        for j in range(num_down_blocks-1,-1,-1):
            if j == 0: # no skip connection at the bottom
                if max_size:
                    num_in = min((2 ** (j+1)) * num_of_filters, 2 * max_size)
                else:
                    num_in = (2 ** (j + 1)) * num_of_filters
                self.h_up.append(ConvBlock(num_in , num_in_filters//2, kernel_size_CNN, num_convolutions_per_block,
                                           is_residual = False))
            else:
                if max_size:
                    num_in = min((2 ** (j+1)) * num_of_filters, 2 * max_size)
                    num_out = min((2 ** (j-1)) * num_of_filters, max_size)
                else:
                    num_in = (2 ** (j + 1)) * num_of_filters
                    num_out = (2 ** (j-1)) * num_of_filters
                self.h_up.append(ConvBlock(num_in, num_out, kernel_size_CNN,num_convolutions_per_block,
                                           is_residual = False))

        self.connections = nn.ModuleList([])
        for k in range(num_down_blocks+1):
            self.connections.append(torch.nn.Identity())

    def forward(self,input, layer_id=-1):
        """Forward pass through the UNet for the on-the-grid CNN

        Args:
            input (tensor): latent representation of the input context (batch, img_width, img_size, num_in_filters)
            layer_id (int), optional: id of the layer to output, by default -1 to return the last one
        Returns:
            tensor: output map of the UNet
        """
        layers = []
        #layers.append(input)
        # Down
        x = input
        for i in range(self.num_down_blocks):
            x = self.h_down[i](x)
            layers.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.h_bottom(x)
        x = self.connections[self.num_down_blocks](x)
        layers.append(x)

        # Up
        for i in range(self.num_down_blocks):

            # upsample
            x = self.upsample(x)

            # pad if necessary and concatenate
            res = layers[self.num_down_blocks - i - 1]
            res = self.connections[self.num_down_blocks-i-1](res)
            h_diff, w_diff = res.shape[-2] - x.shape[-2], res.shape[-1] - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, w_diff, 0, h_diff))
            x = torch.cat([x, res], dim=1)

            # feed through conv block
            x = self.h_up[i](x)
            layers.append(x)

        return layers[layer_id]


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
        dense_layer_widths (list of int): list with the dimensionality of the layers (first entry is the number of input filters (reduced to 1D by average pooling))
    """
    def __init__(self,model, dense_layer_widths):
        super(ConvCNPClassifier,self).__init__()
        self.encoder = model.encoder
        self.CNN = model.CNN

        # add the dense layers
        l = len(dense_layer_widths)
        h = nn.ModuleList([])  # store the layers as a list
        for i in range(0, l - 1):
            h.append(nn.Linear(dense_layer_widths[i],dense_layer_widths[i+1]))
            if i != l - 2:  # no ReLU for the last layer
                h.append(nn.ReLU())
        self.dense_network = nn.Sequential(*h)
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
        x = torch.mean(x,dim=[2,3])
        output_logit = self.dense_network(x)
        output_probs = self.final_activation(output_logit)
        return output_logit, output_probs

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

    def evaluate_accuracy(self,mask,context_img,target_label):
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

class ConvCNPExtractRepresentation(nn.Module):
    """ Modify the convCNP to have a get a representation out

        Args:
            model (nn.module): original ConvCNP
            layer_id (int): id of the layer where the representation is extracted
            pooling (string), optional: type of pooling used to get a 1d representation, one of ["average","max","min","flatten"]
    """
    def __init__(self,model,layer_id, pooling="mean"):
        super(ConvCNPExtractRepresentation, self).__init__()
        self.encoder = model.encoder
        self.CNN = model.CNN
        self.layer_id = layer_id
        self.pooling = pooling

        assert pooling in ["average","max","min","flatten"], "Pooling should be one of " + " ".join(["average","max","min","flatten"])

    def forward(self,mask,context_image):
        """ Foward pass through the ConvCNP model up to the CNN and returning the relevant representation

        Args:
            mask (tensor): binary tensor indicating context pixels with a 1 (batch,img_height,img_width,1)
            context_image (tensor): masked image with 0 everywhere except at context pixels (batch, img_height, img_width, num_input_channels)

        Returns:
            tensor: representation at the corresponding layer (batch, num_features)

        """
        output_encoder = self.encoder(mask,context_image)
        r = self.CNN(output_encoder, layer_id=self.layer_id)
        if self.pooling == "average":
            return torch.mean(r, dim=[-2, -1])
        elif self.pooling == "max":
            return torch.amax(r, dim=[-2, -1])
        elif self.pooling == "min":
            return torch.amin(r, dim=[-2, -1])
        elif self.pooling == "flatten":
            return torch.flatten(r, start_dim=1, end_dim=-1)



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
    num_of_filters_top_UNet =  64
    pooling_size = 2
    max_size = 64
    model = OnTheGridConvCNP(type_CNN=type_CNN,num_input_channels=num_input_channels,num_output_channels=num_output_channels,
                             num_of_filters=num_of_filters,kernel_size_first_convolution=kernel_size_first_convolution,
                             kernel_size_CNN=kernel_size_CNN, num_convolutions_per_block=num_convolutions_per_block,
                             num_dense_layers=num_dense_layers,num_units_dense_layer=num_units_dense_layers,
                             num_residual_blocks=num_residual_blocks, num_down_blocks=num_down_blocks,
                             num_of_filters_top_UNet=num_of_filters_top_UNet, pooling_size=pooling_size,
                             max_size=max_size)
    summary(model, [(1, img_height, img_width), (1, img_height, img_width)])

    # model to extract the representations
    layer_id = 6
    pooling = "max"
    model_r = ConvCNPExtractRepresentation(model,layer_id,pooling)
    summary(model_r, [(1, img_height, img_width), (1, img_height, img_width)])


    # Classfication ConvCNP
    dense_layer_widths = [128,64,64,10]
    classification_model = ConvCNPClassifier(model, dense_layer_widths)
    # freeze the weights from the original CNP
    for param in classification_model.encoder.parameters():
        param.requires_grad = False
    for param in classification_model.CNN.parameters():
        param.requires_grad = False

    summary(classification_model, [(1, img_height, img_width), (1, img_height, img_width)])









