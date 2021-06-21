# Some parts of this code are taken from https://github.com/cambridge-mlg/convcnp.git

import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
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

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])

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

class CNPClassifier(nn.Module):
    """ Modify a CNP to replace the decoder with a classification head
    Args:
        model (nn.module): original CNP
        classification_head_layer_widths (list of int): list with the dimensionality of the layers (first entry is the size of the representation)
    """
    def __init__(self,model,classification_head_layer_widths):
        super(CNPClassifier,self).__init__()
        self.encoder = model.encoder
        l = len(classification_head_layer_widths)
        h = nn.ModuleList([])  # store the hidden layers as a list
        for i in range(0, l - 1):
            h.append(nn.Linear(classification_head_layer_widths[i], classification_head_layer_widths[i + 1]))
            if i != l - 2:  # no ReLU for the last layer
                h.append(nn.ReLU())
        self.classification_head = nn.Sequential(*h)
        self.final_activation = nn.Softmax(dim=-1)

        # for joint training
        self.decoder = model.decoder
        self.loss_unsup = model.loss

    def forward(self,x_context,y_context,joint=False):
        """ Forward pass through the Classification CNP

        Args:
            x_context (tensor): x values of the context points (batch,num_context,input_dim_x)
            y_context (tensor): y values of the context points (batch,num_context,input_dim_y)
            joint (bool): whether it is used for joint training, so if true will return both logits and mean/std, otherwise only logits

        Returns:
            tensor: classification score for the different output classes (batch,num_classes)
            tensor: probability mass for the different output classes (batch,num_classes)
        """
        r = self.encoder(x_context,y_context)
        r = torch.squeeze(r,dim=1)
        output_logit = self.classification_head(r)
        output_probs = self.final_activation(output_logit)

        if joint:
            mean, std = self.decoder(r)
            return output_logit, output_probs, mean, std
        else:
            return output_logit, output_probs

    def loss(self,output_logit,target_label, reduction='mean'):
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        return criterion(output_logit,target_label)

    def joint_loss(self,x_context, y_context, x_target, target_label, y_target, alpha, scale_sup=1, scale_unsup=1):

        # get the output
        output_logit, output_probs, mean, std = self.forward(x_context,y_context,x_target,joint=True)

        # compute the loss
        target_label = target_label.detach().clone()
        select_labelled = target_label != -1
        sup_loss = scale_sup * select_labelled.float() * alpha * self.loss(output_logit, target_label, reduction='none')
        sup_loss = sup_loss.mean()
        unsup_loss = scale_unsup * self.loss_unsup(mean,std,y_target)

        # compute the accuracy
        _, predicted = torch.max(output_probs, dim=1)
        total = (target_label != -1).sum().item()
        if total.item() != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        return sup_loss + unsup_loss, sup_loss.item(), unsup_loss.item(), accuracy, total


    def train_step(self,x_context,y_context,target_label,opt):
        output_logit, output_probs = self.forward(x_context,y_context,joint=False)
        obj = self.loss(output_logit,target_label)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        # return the accuracy as well
        _, predicted = torch.max(output_probs, dim=1)
        total = (target_label != -1).sum().item()
        if total.item() != 0:
            accuracy = ((predicted == target_label).sum()).item() / total
        else:
            accuracy = 0

        return obj.item(), accuracy, total

    def joint_train_step(self,x_context,y_context,target_label,y_target,opt,alpha=1,scale_sup=1,scale_unsup=1):

        obj, sup_loss, unsup_loss, accuracy, total = self.joint_loss(x_context, y_context, x_target, target_label, y_target, alpha, scale_sup=scale_sup, scale_unsup=scale_unsup)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item(), sup_loss.item(), unsup_loss.item(), accuracy, total

    def unsup_train_step(self,x_context,y_context,y_target,opt,l_unsup=1):
        output_logit, _, mean, std = self.forward(x_context,y_context,joint=True)
        obj = l_unsup * self.loss_unsup(mean, std, y_target)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item()  # report the loss as a float

    def evaluate_accuracy(self,x_context,y_context, target_label):
        # compute the logits
        output_logit, output_probs = self.forward(x_context, y_context)

        # get the predictions
        _, predicted = torch.max(output_probs,dim=1)

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

if __name__ == "__main__":
    # CNP
    encoder_layer_widths = [3,128,128,128]
    decoder_layer_widths = [2,128,128,128,2]
    model = CNP(encoder_layer_widths,decoder_layer_widths)
    summary(model, [(10, 2), (10, 1), (100, 2)])

    # Classfication CNP
    classification_head_layer_widths = [128,64,64,10]
    classification_model = CNPClassifier(model,classification_head_layer_widths)
    # freeze the encoder
    for param in classification_model.encoder.parameters():
        param.requires_grad = False
    summary(classification_model, [(784, 2), (784, 1)])

    print("done!")



