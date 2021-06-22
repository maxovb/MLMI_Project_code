# The code was originally copied from https://github.com/cambridge-mlg/convcnp/blob/master/convcnp/set_conv.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torchsummary import summary
from Utils.helper_loss import gaussian_logpdf, mixture_of_gaussian_logpdf

class OffTheGridConvCNP(nn.Module):
    """One-dimensional ConvCNP model.
    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input. Used to discretize function.
        type_CNN (string): one of ["CNN","UNet"]
        num_input_channels (int): number of input channels, i.e. 1 for BW and 3 for RGB
        num_output_channels (int): number of output channels, i.e. 2 (mean+std) for BW and 6 for RGB
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        kernel_size_CNN (int): size of the kernel for the CNN part
        num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
        num_residual_blocks (int): number of residual blocks in the CNN
        num_down_blocks (int): number of down blocks when using UNet
        num_of_filters_top_UNet (int): number of filters for the top UNet layer (doubles all the way down)
        pooling_size (int): pooling size for the UNet
        max_size (int or None): maximum number of features in the UNet
        is_gmm (bool), optional: whether the predictive distribution is a GMM
        classifier_layer_widths (list of int): size of the dense layers in the classifier network in the GMM case
        num_classes (int): number of classes to classify as in the GMM case
        block_center_connections (bool): whether to block the center connections in the UNet
    """

    def __init__(self, learn_length_scale, points_per_unit, type_CNN, num_input_channels, num_output_channels, num_of_filters, kernel_size_CNN, num_convolutions_per_block, num_residual_blocks = None, num_down_blocks=None, num_of_filters_top_UNet=None, pooling_size=None, max_size = None, is_gmm = False, num_classes = None, classifier_layer_widths = None, block_center_connections=False):
        super(OffTheGridConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.multiplier = (2 ** num_down_blocks if type_CNN == "UNet" else 1)
        self.is_gmm = is_gmm
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if is_gmm:
            self.num_classes = num_classes

        if type_CNN == "CNN":
            assert num_residual_blocks, "The argument num_residual blocks should be passed as integer when using the CNN ConvCNP"
            self.CNN = OffTheGridConvCNPCNN(num_of_filters,kernel_size_CNN,num_residual_blocks,num_convolutions_per_block)

        elif type_CNN == "UNet":
            assert num_down_blocks and num_of_filters_top_UNet and pooling_size, "Arguments num_down_blocks, num_of_filters_top_UNet and pooling_size should be passed as integers when using the UNet ConvCNP"
            self.CNN = OffTheGridConvCNPUNet(num_of_filters_top_UNet, num_of_filters, kernel_size_CNN, num_down_blocks, num_convolutions_per_block, pooling_size, max_size, is_gmm, classifier_layer_widths, num_classes, block_center_connections)

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        self.l0 = ConvDeepSet(
            in_channels=num_input_channels,
            out_channels=num_of_filters,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=True
        )
        self.mean_layer = ConvDeepSet(
            in_channels=num_of_filters,
            out_channels=num_output_channels//2,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            in_channels=num_of_filters,
            out_channels=num_output_channels // 2,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

    def forward(self, x, y, x_out):
        """Run the model forward.
        Args:
            x (tensor): Observation locations of shape
                `(batch, data, features)`.
            y (tensor): Observation values of shape
                `(batch, data, outputs)`.
            x_out (tensor): Locations of outputs of shape
                `(batch, data, features)`.
        Returns:
            tuple[tensor]: Means and standard deviations of shape
                `(batch_out, channels_out)`.
        """
        batch_size, num_context_points, num_channels = x.shape[0], x.shape[1], x.shape[2]
        num_target_points = x_out.shape[1]

        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(x_out.shape) == 2:
            x_out = x_out.unsqueeze(2)

        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x).cpu().numpy(),
                    torch.min(x_out).cpu().numpy(), -2.) - 0.1
        x_max = max(torch.max(x).cpu().numpy(),
                    torch.max(x_out).cpu().numpy(), 2.) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(self.device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.l0(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        if self.is_gmm:
            h, logits, probs = self.CNN(h)
            h = h.reshape(h.shape[0] * self.num_classes, h.shape[2], -1).permute(0, 2, 1)
        else:
            h = self.CNN(h)
            h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        if self.is_gmm:
            x_grid = x_grid[:,None,:,:].repeat(1,self.num_classes,1,1)
            x_grid = x_grid.view(batch_size * self.num_classes, x_grid.shape[2], x_grid.shape[3])
            x_out = x_out[:, None, :, :].repeat(1, self.num_classes, 1, 1)
            x_out = x_out.view(batch_size * self.num_classes, x_out.shape[2], x_out.shape[3])

        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))

        if self.is_gmm:
            mean = mean.view(batch_size,self.num_classes,mean.shape[1],mean.shape[2])
            sigma = sigma.view(batch_size, self.num_classes, sigma.shape[1], sigma.shape[2])
            return mean, sigma, logits, probs
        else:
            return mean, sigma

    def loss(self, mean, std, target):
        obj = -gaussian_logpdf(target, mean, std, 'batched_mean')
        return obj

    def train_step(self, x_context, y_context, x_target, target, opt):
        mean, std = self.forward(x_context, y_context, x_target)
        obj = self.loss(mean, std, target)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item()  # report the loss as a float

    def joint_train_step(self, x_context, y_context, x_target, target_label, y_target, opt, alpha=1, scale_sup=1,
                         scale_unsup=1):
        # computing the losses
        obj, sup_loss, unsup_loss, accuracy, total = self.joint_loss(x_context, y_context, x_target, target_label,
                                                                     y_target, alpha=alpha, scale_sup=scale_sup,
                                                                     scale_unsup=scale_unsup)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item(), sup_loss, unsup_loss, accuracy, total

    def joint_loss(self, x_context, y_context, x_target, target_label, y_target, alpha=1, scale_sup=1, scale_unsup=1):
        if self.is_gmm:
            return self.joint_gmm_loss(x_context, y_context, x_target, target_label, y_target, alpha,
                                       scale_sup=scale_sup, scale_unsup=scale_unsup)
        else:
            raise RuntimeError(
                "For joint training of the ConvCNP, use a GMM type, or otherwise use the classifier version")

    def joint_gmm_loss(self, x_context, y_context, x_target, target_label, y_target, alpha=1, scale_sup=1,
                       scale_unsup=1):

        # obtain the batch size
        batch_size = x_context.shape[0]

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
            x_context_unlabelled = x_context[unlabelled_indices]
            y_context_unlabelled = y_context[unlabelled_indices]
            x_target_unlabelled = x_target[unlabelled_indices]
            y_target_unlabelled = y_target[unlabelled_indices]

            # unlabelled batch size
            batch_size_unlabelled = x_context_unlabelled.shape[0]

            # calculate the loss
            unsup_logp = self.unsupervised_gmm_logp(x_context_unlabelled, y_context_unlabelled, x_target_unlabelled,
                                                    y_target_unlabelled)

            J += scale_unsup * unsup_logp
            unsup_loss = - scale_unsup * unsup_logp.item() / batch_size_unlabelled

        if not (all_unlabelled):

            # split to obtain the unlabelled samples
            x_context_labelled = x_context[labelled_indices]
            y_context_labelled = y_context[labelled_indices]
            x_target_labelled = x_target[labelled_indices]
            y_target_labelled = y_target[labelled_indices]
            target_labelled_only = target_label[labelled_indices]

            # unlabelled batch size
            batch_size_labelled = x_context_labelled.shape[0]

            # calculate the loss
            unsup_logp, sup_logp, accuracy, total = self.supervised_gmm_logp(x_context_labelled, y_context_labelled,
                                                                             x_target_labelled, y_target_labelled,
                                                                             target_labelled_only)

            J += scale_unsup * unsup_logp
            unsup_loss = (-J).item() / batch_size
            J += scale_sup * alpha * sup_logp
            sup_loss = scale_sup * (-alpha * sup_logp).item() / batch_size_labelled

        else:
            sup_loss = None
            total = 0
            accuracy = 0

        joint_loss = -J / float(batch_size)

        return joint_loss, sup_loss, unsup_loss, accuracy, total

    def unsupervised_gmm_logp(self, x_context, y_context, x_target, y_target):
        mean, std, logits, probs = self(x_context, y_context, x_target)

        logp = mixture_of_gaussian_logpdf(y_target, mean, std, probs, reduction="sum")

        return logp

    def supervised_gmm_logp(self, x_context, y_context, x_target, y_target, target_label):

        # reconstruction loss
        mean, std, logits, probs = self(x_context, y_context, x_target)

        # repeat the component probabilities to match the shape of the mean and std
        forced_probs = torch.nn.functional.one_hot(target_label.type(torch.int64), num_classes=self.num_classes).to(
                                                   mean.device)

        #  calculate the log-likelihood
        logp = mixture_of_gaussian_logpdf(y_target, mean, std, forced_probs, reduction="sum")

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

    def sample_one_component(self, x_context, y_context, x_target):
        """ Function to pass through model predicting a GMM and outputing the mean and std from one of the component, sampling with a Categorical distribution from the component weights

        Args:
            x_context (tensor): x indices of context points (batch, num_context_points,1)
            y_context (tensor): y indices of context points (batch, num_context_points,1)
            x_target (tensor): x indices of target points (batch, num_target_points,1)

        Return:
            tensor: mean of the sampled component (batch, num_target_points,1)
            tensor: std of the sampled component (batch, num_target_points,1)
        """

        #  get the means, std and weights of the components
        means, stds, logits, probs = self(x_context, y_context, x_target)

        #  sample one of the component
        dist_component = Categorical(probs.type(torch.float))
        sample = dist_component.sample()

        indices = sample[:,None,None,None]
        indices = indices.repeat(1, 1, means.shape[-2], means.shape[-1])
        mean = torch.gather(means, 1, indices)[:, 0, :, :]
        std = torch.gather(stds, 1, indices)[:, 0, :, :]

        return mean, std, probs

    def evaluate_accuracy(self, x_context, y_context, x_target, target_label):

        #  forward pass through the model
        means, stds, logits, probs = self(x_context, y_context, x_target)

        # compute the accuracy
        _, predicted = torch.max(probs, dim=1)
        total = len(target_label)
        if total != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        return accuracy, total

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])

class ConvDeepSet(nn.Module):
    """One-dimensional set convolution layer. Uses an RBF kernel for
    `psi(x, x')`.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        learn_length_scale (bool): Learn the length scales of the channels.
        init_length_scale (float): Initial value for the length scale.
        use_density (bool, optional): Append density channel to inputs.
            Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 init_length_scale,
                 use_density=True):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.use_density = use_density
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        `in_channels + 1`-dimensional representation to dimensionality
        `out_channels`.
        Returns:
            :class:`torch.nn.Module`: Linear layer applied point-wise to
                channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.
        Args:
            x (tensor): Inputs of observations (batch, n, 1)
            y (tensor): Outputs of observations (batch, n, in_channels)
            t (tensor): Inputs to evaluate function at of shape (batch, m, 1)
        Returns:
            tensor: Outputs of evaluated function at z of shape (batch, m, out_channels).
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(t.shape) == 2:
            t = t.unsqueeze(2)

        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        if self.use_density:
            # Compute the extra density channel.
            # Shape: (batch, n_in, 1).
            density = torch.ones(batch_size, n_in, 1).to(x.device)

            # Concatenate the channel.
            # Shape: (batch, n_in, in_channels).
            y_out = torch.cat([density, y], dim=2)
        else:
            y_out = y

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        if self.use_density:
            # Use density channel to normalize convolution
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.
        Args:
            dists (tensor): Pair-wise distances between `x` and `t`.
        Returns:
            tensor: Evaluation of `psi(x, t)` with `psi` an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

class OffTheGridConvCNPCNN(nn.Module):
    """CNN for the on-the-grid version of the Convolutional Conditional Neural Process. See https://arxiv.org/abs/1910.13556 for details.

    Args:
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        kernel_size_CNN (int): size of the kernel for the CNN part
        num_residual_blocks (int): number of residual blocks in the CNN
        num_convolutions_per_block (int): number of convolutional layers per residual blocks in the CNN
    """

    def __init__(self, num_of_filters, kernel_size_CNN, num_residual_blocks, num_convolutions_per_block):
        super(OffTheGridConvCNPCNN, self).__init__()

        # store the layers as a list
        self.h = nn.ModuleList([])
        self.num_residual_blocks = num_residual_blocks
        for i in range(0, num_residual_blocks):
            if i == 0:  # do not use residual blocks for the first block because the number of channel changes
                self.h.append(ConvBlock1D(2 * num_of_filters, num_of_filters, kernel_size_CNN,
                                          num_convolutions_per_block, is_residual=False))
            else:
                self.h.append(ConvBlock1D(num_of_filters, num_of_filters, kernel_size_CNN, num_convolutions_per_block,
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

class OffTheGridConvCNPUNet(nn.Module):
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
        super(OffTheGridConvCNPUNet, self).__init__()

        self.is_gmm = is_gmm
        self.block_center_connections = block_center_connections

        # store some variables
        self.num_down_blocks = num_down_blocks

        self.pool = torch.nn.MaxPool1d(pooling_size)
        self.upsample = torch.nn.Upsample(scale_factor=pooling_size)

        self.h_down = nn.ModuleList([])
        for i in range(num_down_blocks):
            if i == 0:  # do not use residual blocks for the first block because the number of channel changes
                if max_size:
                    num_out = min(num_of_filters,max_size)
                else:
                    num_out = num_of_filters
                self.h_down.append(ConvBlock1D(num_in_filters, num_out, kernel_size_CNN,
                                               num_convolutions_per_block, is_residual=False))
            else:
                if max_size:
                    num_in = min((2**(i-1)) * num_of_filters,max_size)
                    num_out = min((2**(i)) * num_of_filters,max_size)
                else:
                    num_in = (2 ** (i - 1)) * num_of_filters
                    num_out = (2 ** (i)) * num_of_filters
                self.h_down.append(ConvBlock1D(num_in, num_out, kernel_size_CNN,num_convolutions_per_block,
                                               is_residual=False))

        if max_size:
            num = min((2**(num_down_blocks-1)) * num_of_filters, max_size)
        else:
            num = (2**(num_down_blocks-1)) * num_of_filters
        self.h_bottom = ConvBlock1D(num, num, kernel_size_CNN,num_convolutions_per_block, is_residual=False)

        self.h_up = nn.ModuleList([])
        for j in range(num_down_blocks-1,-1,-1):
            if j == 0: # no skip connection at the top
                if max_size:
                    num_in = min((2 ** (j+1)) * num_of_filters, 2 * max_size)
                else:
                    num_in = (2 ** (j + 1)) * num_of_filters
                self.h_up.append(ConvBlock1D(num_in , num_in_filters, kernel_size_CNN, num_convolutions_per_block,
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
                self.h_up.append(ConvBlock1D(num_in, num_out, kernel_size_CNN,num_convolutions_per_block,
                                             is_residual = False))

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
            batch_size, num_extra_dim, num_channels, num_points = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            x = x.view(-1, num_channels, num_points)

        for i in range(self.num_down_blocks):

            # whether this layer has a residual connection or not
            has_residual_connection = (i == self.num_down_blocks-1) or not(self.block_center_connections)

            # upsample
            x = self.upsample(x)

            # pad if necessary and concatenate
            res = layers[self.num_down_blocks - i - 1]
            res = self.connections[self.num_down_blocks - i - 1](res)
            padding_diff = res.shape[-1] - x.shape[-1]
            x = torch.nn.functional.pad(x, (0,padding_diff))

            if has_residual_connection:
                if self.is_gmm:
                    # add the class dimension to the residual connection if GMM type of predictions
                    num_channels_res, res_num_points = res.shape[1], res.shape[2]
                    res = self.expand_repeat_with_all_classes(res)
                    res = res.view(-1, num_channels_res, res_num_points)
                x = torch.cat([x, res], dim=-2)

            # feed through conv block
            x = self.h_up[i](x)

            # reshape to recover the class dimension if GMM type of network
            if self.is_gmm:
                num_channels, num_points = x.shape[1], x.shape[2]
                x_copy = x.view(batch_size, num_extra_dim, num_channels, num_points)
                layers.append(x_copy)
            else:
                layers.append(x)

        # reshape to recover the class dimension if GMM type of network
        if self.is_gmm:
            num_channels, num_points = x.shape[1], x.shape[2]

            x = x.view(batch_size, num_extra_dim, num_channels, num_points)


        return x, layers

    def classify(self,x):
        r = torch.mean(x, dim=-1)
        logits = self.classifier(r)
        probs = self.classifier_activation(logits)
        return logits,probs

    def expand_and_concatenate_with_all_classes(self,x):

        # create the tensor with all classes
        batch_size, num_points = x.shape[0], x.shape[2]
        classes = torch.ones((batch_size,self.num_classes,num_points))
        for i in range(self.num_classes):
            classes[:, i,:] = classes[:, i,:] * i

        # one_hot encoding
        one_hot = torch.nn.functional.one_hot(classes.type(torch.int64), num_classes=self.num_classes).to(x.device)
        one_hot = one_hot.permute(0,1,3,2)

        # repeat x to concatenate to the one_hot encoding
        x = self.expand_repeat_with_all_classes(x)

        # concatenate x and the class one-hot encoding
        x = torch.cat([x, one_hot], dim=2)

        return x

    def expand_repeat_with_all_classes(self,x):
        # repeat x to concatenate to the one_hot encoding
        repeat_size = (1, self.num_classes, 1, 1)
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

class ConvBlock1D(nn.Module):
    """ Convolutional (optionally residual) block for the off-the-grid 1D ConvCNP

    Args:
        num_of_input_channels (int): number of channels at of the input of the block
        num_of_filters (int): number of filters per convolution, i.e. dimension of the output channel size of convolutional layers
        kernel_size (int): size of the kernel
        num_convolutions_per_block (int): number of convolutional layers per residual blocks
        is_residual (bool): whether it is a residual convolutional block or not
    """

    def __init__(self,num_of_input_channels,num_of_filters,kernel_size,num_convolutions_per_block, is_residual):
        super(ConvBlock1D, self).__init__()
        h = nn.ModuleList([])
        for i in range(num_convolutions_per_block):
            if i == 0:
                h.append(DepthwiseSeparableConv1D(num_of_input_channels,num_of_filters,kernel_size))
            else:
                h.append(DepthwiseSeparableConv1D(num_of_filters, num_of_filters, kernel_size))
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


class DepthwiseSeparableConv1D(nn.Module):
    """ Depthwise separable 1D convolution

    Args:
        num_input_channels (int): size of the channel channel dimension of the input
        num_of_filters (int): size of the channel channel dimension of the output
        kernel_size (int): size of the kernel
    """
    def __init__(self,num_input_channels,num_of_filters,kernel_size,enforce_positivity=False,padding=True):
        super(DepthwiseSeparableConv1D,self).__init__()
        if padding:
             padding = kernel_size//2
        else:
            padding = 0

        if enforce_positivity:
            self.depthwise = make_abs_conv(nn.Conv1d)(num_input_channels, num_input_channels, kernel_size=kernel_size,padding=padding,groups=num_input_channels)
            self.pointwise = make_abs_conv(nn.Conv1d)(num_input_channels, num_of_filters, kernel_size=1)
        else:
            self.depthwise = nn.Conv1d(num_input_channels, num_input_channels, kernel_size=kernel_size,padding=padding,groups=num_input_channels)
            self.pointwise = nn.Conv1d(num_input_channels, num_of_filters, kernel_size=1)

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
            return F.conv1d(
                input,
                self.weight.abs(),
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

    return AbsConv

def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.
    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.
    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model

def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.
    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.
    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2

def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.
    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.
    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


if __name__ == "__main__":

    # example data
    from data.GP.GP_data_generator import MultiClassGPGenerator
    import stheno
    train_data = MultiClassGPGenerator([stheno.EQ(),stheno.EQ().periodic(1)], 0.5, kernel_names=["EQ","Periodic"],
                                       batch_size=64, num_tasks=10)
    task, label = train_data.generate_task()
    x_context = task["x_context"]
    y_context = task["y_context"]
    x_target = task["x"]
    y_target = task["y"]

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
    num_of_filters_top_UNet =  64
    pooling_size = 2
    max_size = 64
    model = OffTheGridConvCNP(learn_length_scale = learn_length_scale, points_per_unit=points_per_unit,
                              type_CNN=type_CNN,num_input_channels=num_input_channels,num_output_channels=num_output_channels,
                              num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                              num_convolutions_per_block=num_convolutions_per_block, num_residual_blocks=num_residual_blocks,
                              num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                              pooling_size=pooling_size, max_size=max_size)
    
    summary(model, [x_context.shape[1:], y_context.shape[1:], x_target.shape[1:]])

    opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    model.train_step(x_context,y_context,x_target,y_target,opt)

    # GMM model
    is_gmm = True
    num_classes = 10
    classifier_layer_widths = [64,64,64,10]
    gmm_model = OffTheGridConvCNP(learn_length_scale=learn_length_scale, points_per_unit=points_per_unit,
                              type_CNN=type_CNN, num_input_channels=num_input_channels,
                              num_output_channels=num_output_channels,
                              num_of_filters=num_of_filters, kernel_size_CNN=kernel_size_CNN,
                              num_convolutions_per_block=num_convolutions_per_block,
                              num_residual_blocks=num_residual_blocks,
                              num_down_blocks=num_down_blocks, num_of_filters_top_UNet=num_of_filters_top_UNet,
                              pooling_size=pooling_size, max_size=max_size, is_gmm=is_gmm,
                              classifier_layer_widths=classifier_layer_widths, num_classes=num_classes)

    #summary(gmm_model, [(1, img_height, img_width), (1, img_height, img_width)])
    out = gmm_model(x_context,y_context,x_target)
    mean, std, probs = gmm_model.sample_one_component(x_context,y_context,x_target)

    # define the optimizer
    opt = torch.optim.Adam(gmm_model.parameters(), 1e-4, weight_decay=1e-5)
    joint_loss, sup_loss, unsup_loss, accuracy, total = gmm_model.joint_train_step(x_context, y_context, x_target, label, y_target, opt, alpha=1)
    joint_loss, sup_loss, unsup_loss, accuracy, total = gmm_model.joint_train_step(x_context, y_context, x_target, label, y_target, opt, alpha=1)







