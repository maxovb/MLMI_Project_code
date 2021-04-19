# Some parts of this code are taken from https://github.com/cambridge-mlg/convcnp.git

import torch
from torch import nn
from torchsummary import summary
from Utils.helper_loss import gaussian_logpdf

class CNP(nn.Module):
    """Standard Conditional Neural Process
    See https://arxiv.org/abs/1807.01613 for details.

    Args:
        encoder_layer_widths (list of int): list with the dimensionality of the layers (first entry is input_dim_x+input_dim_y))
        decoder_layer_widths (list of int): list with the dimensionality of the layers (first entry is input_dim_x)
    """
    def __init__(self, encoder_layer_widths, decoder_layer_widths, learning_rate=1e-3):
        super(CNP, self).__init__()
        self.encoder = Encoder(encoder_layer_widths)
        latent_dim = encoder_layer_widths[-1]
        self.decoder = Decoder(decoder_layer_widths,latent_dim)

    def forward(self, x_context, y_context, x_target):
        """Forward pass through the CNP

        Args:
            x_context (tensor): x values of the context points (batch,num_context,input_dim_x)
            y_context (tensor): y values of the context points (batch,num_context,input_dim_y)
            x_target (tensor): x values of the target points (batch,num_target,input_dim_x)
        Returns:
            tensor: predicted mean at every target points (batch, num_target, output_dim)
            tensor: predicted standard deviation at every target points (batch, num_target, output_dim)
        """
        encoder_output = self.encoder(x_context,y_context)
        mean, std = self.decoder(x_target,encoder_output)
        return mean, std

    def loss(self,y_mean,y_std,target):
        obj = -gaussian_logpdf(target, y_mean, y_std, 'batched_mean')
        return obj

    def train_step(self,x_context,y_context,x_target,y_target,opt):
        y_mean, y_std = self.forward(x_context, y_context, x_target)
        obj = self.loss(y_mean,y_std,y_target)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item() # report the loss as a float

class Encoder(nn.Module):
    """Encoder used for standard CNP model.

        Args:
            encoder_layer_widths (list of int): list with the dimensionality of the layers (first entry is input_dim_x+input_dim_y)
    """
    def __init__(self,encoder_layer_widths):
        super(Encoder,self).__init__()
        l = len(encoder_layer_widths)
        h = nn.ModuleList([]) # store the hidden layers as a list
        for i in range(0,l-1):
            h.append(nn.Linear(encoder_layer_widths[i],encoder_layer_widths[i+1]))
            if i != l-2: # no ReLU for the last layer
                h.append(nn.ReLU())

        self.pre_pooling = nn.Sequential(*h)

    def forward(self,x_context,y_context):
        """Forward pass through the encoder

        Args:
            x_context (tensor): context point's x values (batch,num_context,input_dim_x)
            y_context (tensor): context point's y values (batch,num_context,input_dim_y)
        Returns:
            tensor: latent representation of each context (batch,1,latent_dim)
        """

        assert len(x_context.shape) == 3, \
            'Incorrect shapes: ensure x_context is a rank-3 tensor.'
        assert len(y_context.shape) == 3, \
            'Incorrect shapes: ensure y_context is a rank-3 tensor.'

        encoder_input = torch.cat((x_context, y_context), dim=-1)
        x = self.pre_pooling(encoder_input)
        return torch.mean(x, dim=-2, keepdim=True)

class Decoder(nn.Module):
    """Decoder used for standard CNP model.

        Args:
            decoder_layer_widths (list of int): list with the dimensionality of the layers (first entry is the dimension input_dim_x)
            laten_dim (int): size of the latent representation (i.e. output of the encoder)
    """
    def __init__(self,decoder_layer_widths,latent_dim):
        super(Decoder,self).__init__()
        l = len(decoder_layer_widths)
        self.output_dim = decoder_layer_widths[-1]
        h = nn.ModuleList([])  # store the hidden layers as a list
        for i in range(0, l - 1):
            if i == 0:
                h.append(nn.Linear(decoder_layer_widths[i] + latent_dim, decoder_layer_widths[i + 1]))
            else:
                h.append(nn.Linear(decoder_layer_widths[i], decoder_layer_widths[i + 1]))
            if i != l-2: # no ReLU for the last layer
                h.append(nn.ReLU())

        self.post_pooling = nn.Sequential(*h)

    def forward(self, x_target, r):
        """Forward pass through the decoder
        Args:
            x_target (tensor): target locations (batch, num_target, input_dim_x)
            r (tensor): latent representation (batch, 1, latent_dim)

        Returns:
            tensor: predicted mean at every target points (batch, num_target, output_dim)
            tensor: predicted standard deviation at every target points (batch, num_target, output_dim)
        """

        # Reshape inputs to model.
        num_target = x_target.shape[1]

        # If latent representation is global, repeat once for each input.
        if r.shape[1] == 1:
            r = r.repeat(1, num_target, 1)

        x = torch.cat([x_target, r], -1)
        x = self.post_pooling(x)

        mean, std = torch.split(x,self.output_dim//2,dim=-1)
        std = 0.01 + 0.99 * nn.functional.softplus(std)
        return mean,std

if __name__ == "__main__":
    encoder_layer_widths = [3,128,128,128]
    decoder_layer_widths = [1,128,128,128,2]
    model = CNP(encoder_layer_widths,decoder_layer_widths)
    print("done!")
    summary(model,[(10,1),(10,1),(100,1)])



