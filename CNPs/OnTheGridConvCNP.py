import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torchsummary import summary
from Utils.helpers_train import grad_norm_iteration
from Utils.helper_loss import gaussian_logpdf, mixture_of_gaussian_logpdf

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
            is_gmm (bool), optional: whether the predictive distribution is a GMM
            classifier_layer_widths (list of int): size of the dense layers in the classifier network in the GMM case
            num_classes (int): number of classes to classify as in the GMM case

    """
    def __init__(self, type_CNN, num_input_channels, num_output_channels, num_of_filters, kernel_size_first_convolution, kernel_size_CNN, num_convolutions_per_block, num_dense_layers, num_units_dense_layer, num_residual_blocks = None, num_down_blocks=None, num_of_filters_top_UNet=None, pooling_size=None, max_size = None, is_gmm = False, num_classes = None, classifier_layer_widths = None, block_center_connections=False):
        super(OnTheGridConvCNP, self).__init__()

        self.is_gmm = is_gmm

        self.encoder = OnTheGridConvCNPEncoder(num_input_channels,num_of_filters,kernel_size_first_convolution)

        if type_CNN == "CNN":
            assert num_residual_blocks, "The argument num_residual blocks should be passed as integer when using the CNN ConvCNP"
            self.CNN = OnTheGridConvCNPCNN(num_of_filters,kernel_size_CNN,num_residual_blocks,num_convolutions_per_block)

        elif type_CNN == "UNet":
            assert num_down_blocks and num_of_filters_top_UNet and pooling_size, "Arguments num_down_blocks, num_of_filters_top_UNet and pooling_size should be passed as integers when using the UNet ConvCNP"
            self.CNN = OnTheGridConvCNPUNet(num_of_filters_top_UNet, 2 * num_of_filters, kernel_size_CNN, num_down_blocks, num_convolutions_per_block, pooling_size, max_size, is_gmm, classifier_layer_widths, num_classes, block_center_connections)

        self.decoder = OnTheGridConvCNPDecoder(num_of_filters,num_dense_layers, num_units_dense_layer,num_output_channels,is_gmm=is_gmm)

        if self.is_gmm:
            self.num_classes = num_classes

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

        if not(self.is_gmm):
            x = self.CNN(encoder_output)
            mean, std = self.decoder(x)
            return mean, std
        else:
            x, logits, probs = self.CNN(encoder_output)
            mean, std = self.decoder(x)
            return mean, std, logits, probs

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

    def joint_train_step(self, mask, context_img, target_label, target_img, opt,alpha=1,scale_sup=1,scale_unsup=1, consistency_regularization=False, num_sets_of_context=1):
        # computing the losses
        obj, sup_loss, unsup_loss, accuracy, total = self.joint_loss(mask, context_img, target_label, target_img, alpha=alpha, scale_sup=scale_sup, scale_unsup=scale_unsup, consistency_regularization=consistency_regularization, num_sets_of_context=num_sets_of_context)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item(), sup_loss, unsup_loss, accuracy, total
    
    def joint_loss(self,mask,context_img,target_label,target_img,alpha=1,scale_sup=1,scale_unsup=1, consistency_regularization=False, num_sets_of_context=1):
        if self.is_gmm:
            return self.joint_gmm_loss(mask,context_img,target_label,target_img,alpha,scale_sup=scale_sup,scale_unsup=scale_unsup,consistency_regularization=consistency_regularization,num_sets_of_context=num_sets_of_context)
        else:
            raise RuntimeError("For joint training of the ConvCNP, use a GMM type, or otherwise use the classifier version")

    def joint_gmm_loss(self,mask,context_img,target_label,target_img,alpha=1,scale_sup=1,scale_unsup=1, consistency_regularization=False, num_sets_of_context=1):

        # obtain the batch size
        batch_size = mask.shape[0]

        # split into labelled and unlabelled
        labelled_indices = target_label != -1
        unlabelled_indices = torch.logical_not(labelled_indices)

        # check if it is a batch where all samples or all labelled or are all unlabelled
        if torch.all(unlabelled_indices):
            all_unlabelled = True
        else:
            all_unlabelled = False
        if torch.all(labelled_indices):
            all_labelled = True
        else:
            all_labelled = False

        # general objective
        J = 0

        if not (all_labelled):

            # split to obtain the unlabelled samples
            mask_unlabelled = mask[unlabelled_indices]
            context_img_unlabelled = context_img[unlabelled_indices]
            target_img_unlabelled = target_img[unlabelled_indices]

            if len(mask_unlabelled.shape) == 3:
                mask_unlabelled = torch.unsqueeze(mask_unlabelled,dim=1)
                context_img_unlabelled = torch.unsqueeze(context_img_unlabelled, dim=1)
                target_img_unlabelled = torch.unsqueeze(target_img_unlabelled, dim=1)

            # unlabelled batch size
            batch_size_unlabelled = mask_unlabelled.shape[0]

            # calculate the loss
            unsup_logp = self.unsupervised_gmm_logp(mask_unlabelled,context_img_unlabelled,target_img_unlabelled, consistency_regularization, num_sets_of_context)

            J += scale_unsup * unsup_logp
            unsup_loss = - scale_unsup * unsup_logp.item()/batch_size_unlabelled

        if not (all_unlabelled):

            # split to obtain the unlabelled samples
            mask_labelled = mask[labelled_indices]
            context_img_labelled = context_img[labelled_indices]
            target_img_labelled = target_img[labelled_indices]
            target_labelled_only = target_label[labelled_indices]

            if len(mask_labelled.shape) == 3:
                mask_labelled = torch.unsqueeze(mask_labelled,dim=1)
                context_img_labelled = torch.unsqueeze(context_img_labelled, dim=1)
                target_img_labelled = torch.unsqueeze(target_img_labelled, dim=1)

            # unlabelled batch size
            batch_size_labelled = mask_labelled.shape[0]

            # calculate the loss
            unsup_logp, sup_logp, accuracy, total = self.supervised_gmm_logp(mask_labelled, context_img_labelled, target_img_labelled, target_labelled_only)

            J += scale_unsup * unsup_logp
            unsup_loss = (-J).item()/batch_size
            J += scale_sup * alpha * sup_logp
            sup_loss = scale_sup * (-alpha * sup_logp).item() / batch_size_labelled

        else:
            sup_loss = None
            total = 0
            accuracy = 0

        joint_loss = -J/float(batch_size)

        return joint_loss, sup_loss, unsup_loss, accuracy, total

    def unsupervised_gmm_logp(self,mask,context_img,target_img, consistency_regularization=False, num_sets_of_context=1):
        mean, std, logits, probs = self(mask,context_img)

        # permute the tensors to have the number of components as the last batch dimension
        mean = mean.permute(0, 1, 4, 2, 3)
        std = std.permute(0, 1, 4, 2, 3)

        logp = mixture_of_gaussian_logpdf(target_img,mean,std,probs,reduction="sum")

        if consistency_regularization:
            logp += self.gmm_consistency_loss(mean,std,probs,target_img,num_sets_of_context)

        return logp

    def supervised_gmm_logp(self,mask,context_img,target_img,target_label):

        # reconstruction loss
        mean, std, logits, probs = self(mask, context_img)

        # permute the tensors to have the number of components as the last batch dimension
        mean = mean.permute(0,1,4,2,3)
        std = std.permute(0,1,4,2,3)

        # repeat the component probabilities to match the shape of the mean and std
        forced_probs = torch.nn.functional.one_hot(target_label.type(torch.int64), num_classes=self.num_classes).to(mean.device)
        
        # calculate the log-likelihood
        logp = mixture_of_gaussian_logpdf(target_img, mean, std, forced_probs, reduction="sum")

        # classification loss
        criterion = nn.CrossEntropyLoss(reduction="sum")
        classification_logp = - criterion(logits, target_label.type(torch.long))

        # compute the accuracy
        _, predicted = torch.max(probs, dim=1)
        total = len(target_label)
        if total != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        return logp, classification_logp, accuracy, total

    def gmm_consistency_loss(self,mean,std,probs,target_img,num_sets_of_context):
        """ Consistency loss with a GMM predictive, evaluate the liklihood of a prediction with the component weights
            given by another set of context pixels from the same original function

        Args:
            mean (tensor): predicted mean for every component (batch, num_components, num_channels, img_height, img_width)
            std (tensor): predicted std for every component (batch, num_components, num_channels, img_height, img_width)
            probs (tensor): component weight for every GMM component (batch, num_components)
            target_img (tensor): target image to predict (batch, num_channels, img_height, img_width)
            num_sets_of_context (int): number of context sets used for computing the consistency loss

        Returns:

        """
        assert num_sets_of_context == 2, "GMM consistency loss does not handle other number of context sets than 2 at the moment"

        single_set_batch_size = mean.shape[0] / num_sets_of_context
        assert single_set_batch_size == int(single_set_batch_size), "The tensor batch size should be a multiple of the number of sets of context (when using consistency regularization), but got batch size: " + str(mean.shape[0]) + " and num of context sets: " + str(num_sets_of_context)
        single_set_batch_size = int(single_set_batch_size)

        probs_set1, probs_set2 = torch.split(probs, single_set_batch_size, dim=0)
        probs_reordered = torch.cat([probs_set2,probs_set1],dim=0)
        logp = mixture_of_gaussian_logpdf(target_img, mean, std, probs_reordered, reduction="sum")

        return logp

    def sample_one_component(self,mask,context_image):
        """ Function to pass through model predicting a GMM and outputing the mean and std from one of the component, sampling with a Categorical distribution from the component weights

        Args:
            mask (tensor): binary tensor indicating context pixels with a 1 (batch,img_height,img_width,1)
            context_image (tensor): masked image with 0 everywhere except at context pixels (batch, img_height, img_width, num_input_channels)
        
        Return:
            tensor: mean of the sampled component
            tensor: std of the sampled component
        """

        assert is_gmm, "Sampling one component only possible if the model has a GMM predictive"

        # get the means, std and weights of the components
        means, stds, logits, probs = self(mask,context_image)

        # sample one of the component
        dist_component = Categorical(probs.type(torch.float))
        sample = dist_component.sample()

        indices = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sample,dim=-1),dim=-1),dim=-1),dim=-1)
        indices = indices.repeat(1,1,means.shape[-3],means.shape[-2],means.shape[-1])
        mean = torch.gather(means,1,indices)[:,0,:,:,:]
        std = torch.gather(stds, 1, indices)[:,0,:,:,:]

        return mean, std, probs, sample

    def evaluate_accuracy(self, mask,context_image,target_label):

        # forward pass through the model
        means, stds, logits, probs = self(mask,context_image)

        # compute the accuracy
        _, predicted = torch.max(probs, dim=1)
        total = len(target_label)
        if total != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        return accuracy, total

    def get_last_shared_layer(self):
        return self.CNN.get_last_shared_layer()

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

    def get_last_shared_layer(self):
        return self.h[-1]

class OnTheGridConvCNPUNet(nn.Module):
    """U-Net for the CNN part of the on-the-grid version of the Convolutional Conditional Neural Process.

        Args:
            num_in_filters (int): number of filters taken in the network
            num_of_filters (int): number of filters per convolution at the top, doubles at every step down, and divides by two at every step up
            kernel_size_CNN (int): size of the kernel
            num_down_blocks (int): number of blocks until the bottleneck
            num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
            pooling_size (int): size of the maxpooling layers
            is_gmm (bool): whether the predictive distribution is a GMM
            classifier_layer_widths (list of int): widht of the classification layers
            num_classes (int): number of classes to classify as
    """
    def __init__(self, num_of_filters, num_in_filters, kernel_size_CNN, num_down_blocks, num_convolutions_per_block, pooling_size, max_size=None, is_gmm = False, classifier_layer_widths = None, num_classes=None, block_center_connections=False):
        super(OnTheGridConvCNPUNet, self).__init__()

        self.is_gmm = is_gmm
        self.block_center_connections = block_center_connections

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
            if j == 0: # no skip connection at the top
                if max_size:
                    num_in = min((2 ** (j+1)) * num_of_filters, 2 * max_size)
                else:
                    num_in = (2 ** (j + 1)) * num_of_filters
                self.h_up.append(ConvBlock(num_in , num_in_filters//2, kernel_size_CNN, num_convolutions_per_block,
                                           is_residual = False))
            else:
                has_residual_connection = (j == 0) or not(block_center_connections)
                if max_size:
                    if has_residual_connection:
                        num_in = min((2 ** (j+1)) * num_of_filters, 2 * max_size) + (num_classes if (is_gmm and j == num_down_blocks-1) else 0)
                        num_out = min((2 ** (j-1)) * num_of_filters, max_size)
                    else:
                        num_in = min((2 ** (j)) * num_of_filters, max_size) + (num_classes if (is_gmm and j == num_down_blocks - 1) else 0)
                        num_out = min((2 ** (j - 1)) * num_of_filters, max_size)
                else:
                    if has_residual_connection:
                        num_in = (2 ** (j + 1)) * num_of_filters + (num_classes if (is_gmm and j == num_down_blocks-1) else 0)
                        num_out = (2 ** (j-1)) * num_of_filters
                    else:
                        num_in = (2 ** (j)) * num_of_filters + (num_classes if (is_gmm and j == num_down_blocks - 1) else 0)
                        num_out = (2 ** (j - 1)) * num_of_filters
                self.h_up.append(ConvBlock(num_in, num_out, kernel_size_CNN,num_convolutions_per_block,is_residual = False))

        self.connections = nn.ModuleList([])
        for k in range(num_down_blocks+1):
            self.connections.append(torch.nn.Identity())

        if is_gmm:
            assert num_classes and classifier_layer_widths, "num classes and classifier layers widths should be defined if using the GMM version of the UNetCNP"
            self.num_classes = num_classes

            h_classifier = nn.ModuleList([])
            l = len(classifier_layer_widths)
            for i in range(len(classifier_layer_widths)-1):
                h_classifier.append(nn.Linear(classifier_layer_widths[i], classifier_layer_widths[i + 1]))
                if i < l - 2:
                    h_classifier.append(nn.ReLU())
            self.classifier = nn.Sequential(*h_classifier)
            self.classifier_activation = nn.Softmax(dim=-1)


    def down(self,input,layers=None):

        if not(layers):
            layers = []

        # Down
        x = input
        for i in range(self.num_down_blocks):
            x = self.h_down[i](x)
            layers.append(x)
            x = self.pool(x)
        return x, layers

    def up(self,input,layers=None):

        if not(layers):
            layers = []

        # Up
        x = input

        # reshape to have only one batch dimension if GMM type of network
        if self.is_gmm:
            batch_size, num_extra_dim, num_channels, img_height, img_width = x.shape[0], x.shape[1], x.shape[2], \
                                                                             x.shape[3], x.shape[4]
            x = x.view(-1, num_channels, img_height, img_width)

        for i in range(self.num_down_blocks):

            # whether this layer has a residual connection or not
            has_residual_connection = (i == self.num_down_blocks-1) or not(self.block_center_connections)

            # upsample
            x = self.upsample(x)

            # pad if necessary and concatenate
            res = layers[self.num_down_blocks - i - 1]
            res = self.connections[self.num_down_blocks - i - 1](res)
            h_diff, w_diff = res.shape[-2] - x.shape[-2], res.shape[-1] - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, w_diff, 0, h_diff))

            if has_residual_connection:
                if self.is_gmm:
                    # add the class dimension to the residual connection if GMM type of predictions
                    num_channels_res, res_height, res_width = res.shape[1], res.shape[2], res.shape[3]
                    res = self.expand_repeat_with_all_classes(res)
                    res = res.view(-1, num_channels_res, res_height, res_width)
                x = torch.cat([x, res], dim=-3)

            # feed through conv block
            x = self.h_up[i](x)

            # reshape to recover the class dimension if GMM type of network
            if self.is_gmm:
                num_channels, img_height, img_width = x.shape[1], x.shape[2], x.shape[3]
                x_copy = x.view(batch_size, num_extra_dim, num_channels, img_height, img_width)
                layers.append(x_copy)
            else:
                layers.append(x)

        # reshape to recover the class dimension if GMM type of network
        if self.is_gmm:
            num_channels, img_height, img_width = x.shape[1], x.shape[2], x.shape[3]

            x = x.view(batch_size, num_extra_dim, num_channels, img_height, img_width)


        return x, layers

    def classify(self,x):
        r = torch.mean(x, dim=(-2, -1))
        logits = self.classifier(r)
        probs = self.classifier_activation(logits)
        return logits,probs

    def expand_and_concatenate_with_all_classes(self,x):

        # create the tensor with all classes
        batch_size, img_height, img_width = x.shape[0], x.shape[2],x.shape[3]
        classes = torch.ones((batch_size,self.num_classes,img_height,img_width))
        for i in range(self.num_classes):
            classes[:, i,:,:] = classes[:, i,:,:] * i

        # one_hot encoding
        one_hot = torch.nn.functional.one_hot(classes.type(torch.int64), num_classes=self.num_classes).to(x.device)
        one_hot = one_hot.permute(0,1,4,2,3)

        # repeat x to concatenate to the one_hot encoding
        x = self.expand_repeat_with_all_classes(x)

        # concatenate x and the class one-hot encoding
        x = torch.cat([x, one_hot], dim=2)

        return x

    def expand_repeat_with_all_classes(self,x):
        # repeat x to concatenate to the one_hot encoding
        repeat_size = (1, self.num_classes, 1, 1, 1)
        x = torch.unsqueeze(x, dim=1).repeat(repeat_size)
        return x

    def forward(self,input, layer_id=-1):
        """Forward pass through the UNet for the on-the-grid CNN

        Args:
            input (tensor): latent representation of the input context (batch, img_width, img_size, num_in_filters)
            layer_id (int), optional: id of the layer to output, by default -1 to return the last one
        Returns:
            tensor: output map of the UNet
        """
        layers = []
        x, layers = self.down(input, layers=layers)

        # Bottleneck
        x = self.h_bottom(x)
        x = self.connections[self.num_down_blocks](x)

        # classifier if GMM type
        if self.is_gmm:
            # classify
            logits, probs = self.classify(x)

            # expand and concatenate with the possible classes
            x = self.expand_and_concatenate_with_all_classes(x)

        layers.append(x)

        x, layers= self.up(x,layers)

        if not(self.is_gmm):
            return layers[layer_id]
        else:
            return layers[layer_id], logits, probs

    def get_last_shared_layer(self):
        return self.h_bottom


class OnTheGridConvCNPDecoder(nn.Module):
    """Decoder for the on-the-grid version of the Convolutional Conditional Neural Process. See https://arxiv.org/abs/1910.13556 for details.

    Args:
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        num_dense_layers (int): number of dense layers at the end
        num_units_dense_layer (int): number of nodes in the hidden dense layers
        num_output_channels (int): number of output channels, i.e. 2 (mean+std) for BW and 3 for RGB
    """

    def __init__(self,num_of_filters,num_dense_layers, num_units_dense_layer,num_output_channels, is_gmm=False):
        super(OnTheGridConvCNPDecoder, self).__init__()

        self.is_gmm = is_gmm
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
        x = input

        # reshape to have only one batch dimension if GMM type of network
        if self.is_gmm:
            batch_size, num_extra_dim, num_channels, img_height, img_width = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            x = x.view(-1, num_channels, img_height, img_width)

        # pass through the network
        x = self.dense_network(x)

        # reshape to recover the class dimension if GMM type of network
        if self.is_gmm:
            num_channels = x.shape[1]
            x = x.view(batch_size, num_extra_dim, num_channels, img_height, img_width)

        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
        elif len(x.shape) == 5:
            x = x.permute(0, 1, 3, 4, 2)
        else:
            raise RuntimeError("x shape at the output of the encoder is invalid: " + str(x.shape))

        mean, std = torch.split(x, self.num_output_channels // 2, dim=-1)
        std = 0.01 + 0.99 * nn.functional.softplus(std)

        return mean, std


class ConvCNPClassifier(nn.Module):
    """ Modify a CNP to replace the decoder with a classification head
    Args:
        model (nn.module): original CNP
        dense_layer_widths (list of int): list with the dimensionality of the layers (first entry is the number of input filters (reduced to 1D by average pooling))
        layer_id (int): id of the layer from which to extract the representation
        pooling (string): type of pooling to perform, one of ["average", "max", "min"]
        dropout (bool, optional): whether to use dropout between the dense layers
    """
    def __init__(self,model, dense_layer_widths, layer_id=-1, pooling="average", dropout=False):
        super(ConvCNPClassifier,self).__init__()
        self.encoder = model.encoder
        self.CNN = model.CNN
        self.decoder = model.decoder
        self.loss_unsup = model.loss
        self.layer_id = layer_id
        self.pooling = pooling

        self.is_gmm = False

        self.task_weights = torch.nn.Parameter(torch.ones(2).float(), requires_grad=True)

        # add the dense layers
        l = len(dense_layer_widths)
        h = nn.ModuleList([])  # store the layers as a list
        for i in range(0, l - 1):
            h.append(nn.Linear(dense_layer_widths[i],dense_layer_widths[i+1]))
            if i != l - 2:  # no ReLU for the last layer
                h.append(nn.ReLU())
                if dropout:
                    h.append(nn.Dropout(0.5))
        self.dense_network = nn.Sequential(*h)
        self.final_activation = nn.Softmax(dim=-1)


    def forward(self,mask,context_img, joint=False):
        """ Forward pass through the Classification CNP

        Args:
            mask (tensor): binary mask locating context pixels (batch,img_height,img_width,1)
            context_img (tensor): context pixels with non-context points masked (batch,img_height,img_width,num_channels)
            joint (bool): whether it is used for joint training, so if true will return both logits and mean/std, otherwise only logits

        Returns:
            tensor: classification score for the different output classes (batch,num_classes)
            tensor: probability mass for the different output classes (batch,num_classes)
        """
        output_encoder = self.encoder(mask,context_img)

        # supervised part
        x = self.CNN(output_encoder, layer_id=self.layer_id)
        if self.pooling == "average":
            x = torch.mean(x,dim=[2,3])
        elif self.pooling == "max":
            x = torch.amax(x, dim=[2, 3])
        elif self.pooling == "min":
            x = torch.amin(x, dim=[2, 3])
        output_logit = self.dense_network(x)
        output_probs = self.final_activation(output_logit)

        if joint:
            # unsupervised part
            x = self.CNN(output_encoder, layer_id=-1)
            mean, std = self.decoder(x)
            return output_logit, output_probs, mean, std
        else:
            return output_logit, output_probs

    def loss(self,output_logit,target_label, reduction='mean'):
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        loss = criterion(output_logit,target_label)
        return loss

    def joint_loss(self,mask,context_img,target_label,target_image,alpha=1,scale_sup=1,scale_unsup=1,consistency_regularization=False,num_sets_of_context=1,grad_norm_iterator=None):

        # obtain the predictions
        output_logit, output_probs, mean, std = self(mask,context_img,joint=True)

        # pre-allocate variables:
        unsup_task_loss = []
        sup_task_loss = []

        # compute the losses
        target_label_for_evaluating = target_label.clone().detach()
        select_labelled = target_label != -1
        target_label_for_evaluating[torch.logical_not(select_labelled)] = 0
        sup_loss = scale_sup * select_labelled.float() * alpha * self.loss(output_logit,target_label_for_evaluating, reduction='none')
        sup_loss = sup_loss.mean()
        rec_loss = scale_unsup * self.loss_unsup(mean,std,target_image)

        # append to the list of tasks loss
        unsup_task_loss.append(rec_loss)
        sup_task_loss.append(sup_loss)

        if consistency_regularization:
            cons_loss = scale_unsup * self.consistency_loss(output_logit, num_sets_of_context)
            unsup_task_loss.append(cons_loss)

        task_loss = torch.stack(unsup_task_loss + sup_task_loss)
        unsup_task_loss = torch.stack(unsup_task_loss)
        sup_task_loss = torch.stack(sup_task_loss)

        # return the accuracy as well
        _, predicted = torch.max(output_probs, dim=1)
        total = (target_label != -1).sum().item()
        if total != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        if not (hasattr(self, "task_weights")):
            self.task_weights = torch.ones(len(task_loss),device=mean.device).float()

        # weights
        n_unsup = len(unsup_task_loss)
        self.task_weights_unsup = self.task_weights[:n_unsup]
        self.task_weights_sup = self.task_weights[n_unsup:]

        unsup_loss = torch.sum(unsup_task_loss)
        sup_loss = torch.sum(sup_task_loss)
        joint_loss = torch.sum(task_loss)

        obj = torch.sum(torch.mul(self.task_weights, task_loss))

        if grad_norm_iterator:
            grad_norm_iterator.store_norm(task_loss)

        return obj, joint_loss.item(), sup_loss.item(), unsup_loss.item(), accuracy, total

    def consistency_loss(self,output_logit, num_sets_of_context=1):

        # obtain the probability distribution
        probs = nn.Softmax(dim=-1)(output_logit)

        assert num_sets_of_context == 2, "Consistency loss does not handle other number of context sets than 2 at the moment"

        # get the original batch size
        single_set_batch_size = output_logit.shape[0] / num_sets_of_context
        assert single_set_batch_size == int(single_set_batch_size), "The tensor batch size should be a multiple of the number of sets of context (when using consistency regularization), but got batch size: " + str(mean.shape[0]) + " and num of context sets: " + str(num_sets_of_context)
        single_set_batch_size = int(single_set_batch_size)

        # split between the two sets of context sets
        probs_set1, probs_set2 = torch.split(probs, single_set_batch_size, dim=0)

        # compute the Jensen Shannon divergence
        m = probs_set1 + probs_set2
        loss = 0.0
        dist1 = Categorical(probs_set1)
        dist2 = Categorical(probs_set2)
        distm = Categorical(m)
        loss += kl_divergence(dist1,distm)
        loss += kl_divergence(dist2,distm)
        loss = 0.5 * torch.mean(loss)

        return loss

    def train_step(self,mask,context_img,target_label,opt):
        output_logit, output_probs = self.forward(mask,context_img,joint=False)
        obj = self.loss(output_logit,target_label)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        # return the accuracy as well
        _, predicted = torch.max(output_probs, dim=1)
        total = (target_label != -1).sum().item()
        if total != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        return obj.item(), accuracy, total

    def joint_train_step(self,mask,context_img,target_label,target_image,opt,alpha=1, scale_sup=1, scale_unsup=1,consistency_regularization=False,num_sets_of_context=1, grad_norm_iterator=None):

        obj, joint_loss, sup_loss, unsup_loss, accuracy, total = self.joint_loss(mask,context_img,target_label,target_image,alpha=alpha,scale_sup=scale_sup,scale_unsup=scale_unsup,consistency_regularization=consistency_regularization,num_sets_of_context=num_sets_of_context,grad_norm_iterator=grad_norm_iterator)

        # optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return joint_loss, sup_loss, unsup_loss, accuracy, total

    def unsup_train_step(self,mask,context_img,target_image,opt,l_unsup=1):
        output_logit, _, mean, std = self.forward(mask, context_img, joint=True)
        obj = l_unsup * self.loss_unsup(mean, std, target_image)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item()  # report the loss as a float

    def evaluate_accuracy(self,mask,context_img,target_label):
        # compute the logits
        output_logit, output_probs = self.forward(mask,context_img)

        # get the predictions
        _, predicted = torch.max(output_probs,dim=1)

        # get the total number of labels
        total = target_label.size(0)

        # compute the accuracy
        accuracy = ((predicted == target_label).sum()).item()/total

        return accuracy, total

    def get_last_shared_layer(self):
        return self.CNN.get_last_shared_layer()

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
    """
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
    pooling = "average"
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
    """
    # GMM model
    is_gmm = True
    num_classes = 10
    classifier_layer_widths = [64,64,64,10]
    gmm_model = OnTheGridConvCNP(type_CNN=type_CNN, num_input_channels=num_input_channels,
                             num_output_channels=num_output_channels,
                             num_of_filters=num_of_filters, kernel_size_first_convolution=kernel_size_first_convolution,
                             kernel_size_CNN=kernel_size_CNN, num_convolutions_per_block=num_convolutions_per_block,
                             num_dense_layers=num_dense_layers, num_units_dense_layer=num_units_dense_layers,
                             num_residual_blocks=num_residual_blocks, num_down_blocks=num_down_blocks,
                             num_of_filters_top_UNet=num_of_filters_top_UNet, pooling_size=pooling_size,
                             max_size=max_size, is_gmm=is_gmm, classifier_layer_widths=classifier_layer_widths,
                             num_classes=num_classes)
    #summary(gmm_model, [(1, img_height, img_width), (1, img_height, img_width)])
    mask = torch.randn((6, 1, img_height, img_width))
    context_img = torch.randn((6, 1, img_height, img_width))
    target_img = torch.randn((6, 1, img_height, img_width))
    target_label = torch.ones((6,1))
    out = gmm_model(mask,context_img)
    mean, std, probs = gmm_model.sample_one_component(mask,context_img)

    # define the optimizer
    opt = torch.optim.Adam(gmm_model.parameters(), 1e-4, weight_decay=1e-5)
    joint_loss, sup_loss, unsup_loss, accuracy, total = gmm_model.joint_train_step(mask, context_img, target_label, target_img, opt, alpha=1)
    joint_loss, sup_loss, unsup_loss, accuracy, total = gmm_model.joint_train_step(mask, context_img, target_label, target_img, opt, alpha=1)








