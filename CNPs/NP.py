# Some parts of this code are taken from https://github.com/cambridge-mlg/convcnp.git

import torch
import time
from torch import nn
from torch.distributions import Categorical, Normal
from torch.distributions.kl import kl_divergence
from torchsummary import summary
from Utils.helper_loss import gaussian_logpdf


class NP(nn.Module):
    """Standard Neural Process
    See https://arxiv.org/abs/1807.01622 for details.

    Args:
        encoder_layer_widths (list of int): list with the dimensionality of the layers (first entry is input_dim_x+input_dim_y))
        decoder_layer_widths (list of int): list with the dimensionality of the layers (first entry is input_dim_x)
        classifier_layer_widths (list of int): list with teh dimensionality of the layers from the aggregate r to the classification output
        latent_network_layer_widths (list of int): list with teh dimensionality of the layers form the aggregate r to the continuous latent parameters
        prior (string), optional: type of prior used on the latent variables
    """

    def __init__(self, encoder_layer_widths, decoder_layer_widths, classifier_layer_widths, latent_network_layer_widths, prior="UnitGaussian"):
        super(NP, self).__init__()
        self.num_classes = classifier_layer_widths[-1]
        self.latent_dim = latent_network_layer_widths[-1]//2

        self.encoder = Encoder(encoder_layer_widths)
        self.classifier = Classifier(classifier_layer_widths)
        self.latent_network = LatentNetwork(latent_network_layer_widths)
        self.sampler = GaussianSampler()
        self.decoder = Decoder(decoder_layer_widths, self.latent_dim + self.num_classes)

        if prior == "UnitGaussian":
            self.prior = Normal(loc=torch.zeros(self.latent_dim),scale=torch.ones(self.latent_dim))

    def forward(self, x_context, y_context, x_target, num_samples=1):
        """Forward pass through the NP

                Args:
                    x_context (tensor): x values of the context points (batch,num_context,input_dim_x)
                    y_context (tensor): y values of the context points (batch,num_context,input_dim_y)
                    x_target (tensor): x values of the target points (batch,num_target,input_dim_x)
                    num_samples (tensor), optional, optional: number of samples for every batch element
                Returns:
                    tensor: predicted mean at every target points (batch * num_samples, num_target, output_dim)
                """

        # encoder
        r = self.encoder(x_context,y_context)

        # class latent variable
        logits, probs = self.classifier(r)
        dist = Categorical(probs)
        sampled_classes = dist.sample((num_samples,))
        sampled_classes = sampled_classes.reshape(-1)
        one_hot = torch.nn.functional.one_hot(sampled_classes, num_classes=self.num_classes).to(r.device)

        # continuous latent variable
        mean, std = self.latent_network(torch.cat(num_samples * [r], dim = 0),one_hot)

        # sample from the continuous latent
        z = self.sampler(mean, std)

        # output
        y_prediction = self.decoder(torch.cat(num_samples * [x_target],dim=0), one_hot, z)

        return y_prediction

    def joint_train_step(self,x_context,y_context,x_target,target_label,y_target,opt,alpha,num_samples_expectation=16, std_y=0.1, parallel=False):

        obj, sup_loss, unsup_loss, accuracy, total = self.joint_loss(x_context,y_context,x_target,target_label,y_target,alpha,num_samples_expectation,std_y,parallel)

        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()

        return obj.item(), sup_loss, unsup_loss, accuracy, total

    def joint_loss(self,x_context,y_context,x_target,target_label,y_target,alpha,num_samples_expectation=16, std_y=0.1, parallel=False):

        # obtain the batch size
        batch_size = x_context.shape[0]

        # split into labelled and unlabelled
        labelled_indices = target_label != -1
        unlabelled_indices = torch.logical_not(labelled_indices)

        # check if there are labelled samples
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

        if not(all_labelled):
            # split to obtain the unlabelled samples
            x_context_unlabelled = x_context[unlabelled_indices]
            y_context_unlabelled = y_context[unlabelled_indices]
            x_target_unlabelled = x_target[unlabelled_indices]
            y_target_unlabelled = y_target[unlabelled_indices]

            # unlabelled batch size
            batch_size_unlabelled = x_context_unlabelled.shape[0]

            # unlabelled objective
            U = self.unlabelled_objective(x_context_unlabelled, y_context_unlabelled, x_target_unlabelled,
                                        y_target_unlabelled, num_samples_expectation, std_y, parallel)

            # general objective
            J += torch.sum(U) 
            unsup_loss = -J.item()/batch_size_unlabelled

        # obtain the objective and classification loss for labelled samples
        if not(all_unlabelled):
            x_context_labelled = x_context[labelled_indices]
            y_context_labelled = y_context[labelled_indices]
            x_target_labelled = x_target[labelled_indices]
            y_target_labelled = y_target[labelled_indices]
            target_labelled_only = target_label[labelled_indices]

            # labelled batch size
            batch_size_labelled = x_context_labelled.shape[0]

            # labelled objective
            L = self.labelled_objective(x_context_labelled, y_context_labelled, x_target_labelled, y_target_labelled,
                                        target_labelled_only, num_samples_expectation, std_y)
            
            # update the general loss
            J = J + torch.sum(L)
            unsup_loss = -J.item()/(batch_size)

            # classification loss
            r = self.encoder(x_context_labelled,y_context_labelled)
            logits, probs = self.classifier(r)
            criterion = nn.CrossEntropyLoss(reduction="sum")
            classification_loss = alpha * criterion(logits,target_labelled_only.type(torch.long))
            J = J - classification_loss

            sup_loss = classification_loss.item()/batch_size_labelled

            # return the accuracy as well
            _, predicted = torch.max(probs, dim=1)
            total = len(target_labelled_only)
            if total != 0:
                accuracy = ((predicted == target_labelled_only).sum()).item() / total
            else:
                accuracy = 0
        else:
            sup_loss = None
            total = 0
            accuracy = 0

        # loss to minimize
        obj = -J/batch_size

        return obj, sup_loss, unsup_loss, accuracy, total

    def unlabelled_objective(self, x_context_unlabelled, y_context_unlabelled, x_target_unlabelled, y_target_unlabelled, num_samples_expectation=16, std_y = 0.1, parallel=False):
        """ Evaluate the unlabelled objective, by marginalizing over the class latent variable

        Args:
            x_context_unlabelled (tensor): x values of the context points (batch,num_context,x_dim)
            y_context_unlabelled (tensor): y values of the context points (batch,num_context,y_dim)
            x_target_unlabelled (tensor): x values of the target points (batch,num_target,x_dim)
            y_target_unlabelled (tensor): y values of the target points (batch,num_target,y_dim)
            num_samples_expectation (int): number of samples for the monte carlo approximation of the expectation of the reconstruction loss
            std_y (int), optional: standard deviation of the likelihood (default 0.1)
            summing (bool), optional: whether the marginalization should be done in parallel or not (parallel is faster but takes more GPU memory)
        Returns:
            (tensor): value of the unlabelled objective
        """

        # encoder
        r = self.encoder(x_context_unlabelled, y_context_unlabelled)

        # classifier
        logits, probs = self.classifier(r)

        # propagate with all classes
        batch_size = r.shape[0]
        list_classes = []
        U = 0
        if parallel:
            classes = torch.ones((batch_size, self.num_classes),device=r.device)
            for i in range(self.num_classes):
                classes[:,i] = classes[:,i] * i
            L = self.labelled_objective(x_context_unlabelled,y_context_unlabelled,x_target_unlabelled,y_target_unlabelled,classes,num_samples_expectation,std_y,r=r)
            U += torch.sum(probs * L,dim=-1)
            
        else:
            for i in range(self.num_classes):
                classes = torch.ones(batch_size) * i
                L = self.labelled_objective(x_context_unlabelled,y_context_unlabelled,x_target_unlabelled,y_target_unlabelled,classes,num_samples_expectation,std_y,r=r)
                U += probs[:,i] * L
        H = -torch.sum(probs * torch.log(probs), dim=1) # entropy
        U += H

        return U


    def labelled_objective(self,x_context_labelled,y_context_labelled,x_target_labelled,y_target_labelled,class_labels,num_samples_expectation,std_y=0.1,r=None):
        """ Evaluate the labelled objective, so with the class label given

            Args:
                x_context_labelled (tensor): x values of the context points (batch,num_context,x_dim)
                y_context_labelled (tensor): y values of the context points (batch,num_context,y_dim)
                x_target_labelled (tensor): x values of the target points (batch,num_target,x_dim)
                y_target_labelled (tensor): y values of the target points (batch,num_target,y_dim)
                class_labels (tensor): class latent variables (batch)
                num_samples_expectation (int): number of samples for the monte carlo approximation of the expectation of the reconstruction loss
                std_y (int), optional: standard deviation of the likelihood (default 0.1)
                r (tensor, optional: embedding of the context points, i.e. output of the encoder (batch,r_dim)
            Returns:
                (tensor): value of the labelled objective
            """
        r_orig = r
        if r == None:
            r = self.encoder(x_context_labelled,y_context_labelled)

        # one hot encoding of the class labels
        one_hot = torch.nn.functional.one_hot(class_labels.type(torch.int64), num_classes=self.num_classes).to(r.device)

        # if running the marginalization over classes in parallel extend the representation
        if one_hot.shape[:-1] != r.shape[:-1]:
            r = torch.unsqueeze(r,dim=1)
            r = r.repeat(1,self.num_classes,1)
            x_target_labelled = torch.unsqueeze(x_target_labelled,dim=1)
            x_target_labelled = x_target_labelled.repeat(1,self.num_classes,1,1)
            y_target_labelled = torch.unsqueeze(y_target_labelled ,dim=1)
            y_target_labelled = y_target_labelled.repeat(1,self.num_classes,1,1)

        # get the parameter of the distribution over the continuous latent variables
        mean_latent, std_latent = self.latent_network(r, one_hot)

        # compute the KL divergence
        try:
            posterior = Normal(loc=mean_latent, scale=std_latent)
        except ValueError:
            with open("error.txt","w") as f:
                f.write("-------------- x_context_labelled ------------------")
                f.write(str(x_context_labelled))
                f.write("\n")
                f.write("-------------- y_context_labelled ------------------")
                f.write(str(y_context_labelled))
                f.write("\n")
                f.write("-------------- r_orig ------------------")
                f.write(str(r_orig))
                f.write("\n")
                f.write("-------------- r ------------------")
                f.write(str(r))
                f.write("\n")
                f.write("-------------- one_hot ------------------")
                f.write(str(one_hot))
                f.write("\n")
                f.write("-------------- loc ------------------")
                f.write(str(mean_latent))
                f.write("\n")
                f.write("-------------- scale ------------------")
                f.write(str(std_latent))
                f.write("\n")
            posterior = Normal(loc=mean_latent, scale=std_latent)

        kl = kl_divergence(posterior,self.prior)

        # sample from the contiuous latent distribution
        list_samples = [torch.unsqueeze(self.sampler(mean_latent, std_latent),dim=-2) for i in range(num_samples_expectation)]
        z = torch.cat(list_samples, dim=-2)

        # pass through the decoder
        one_hot_repeated = torch.cat(num_samples_expectation * [torch.unsqueeze(one_hot,dim=-2)], dim=-2)
        x_target_labelled_repeated = torch.cat(num_samples_expectation * [torch.unsqueeze(x_target_labelled,dim=-3)], dim=-3)
        output = self.decoder(x_target_labelled_repeated, z, one_hot_repeated)

        # compute the likelihood
        y_target_labelled_repeated = torch.cat(num_samples_expectation * [torch.unsqueeze(y_target_labelled,dim=-3)], dim=-3)
        start_idx_sum = -2
        likelihood = gaussian_logpdf(y_target_labelled_repeated, output, std_y, 'samples_mean', start_idx_sum)

        return likelihood - kl.sum(dim=-1)

class Encoder(nn.Module):
    """Encoder used for standard NP model.

        Args:
            encoder_layer_widths (list of int): list with the dimensionality of the layers up to the representation r (first entry is input_dim_x+input_dim_y)
    """

    def __init__(self,encoder_layer_widths):
       super(Encoder,self).__init__()
       l = len(encoder_layer_widths)
       h = nn.ModuleList([]) # store the hidden layers as a list
       for i in range(0,l-1):
           h.append(nn.Linear(encoder_layer_widths[i],encoder_layer_widths[i+1]))
           h.append(nn.ReLU())
       self.pre_pooling = nn.Sequential(*h)

    def forward(self,x_context,y_context):
        """Forward pass through the encoder

        Args:
            x_context (tensor): context point's x values (batch,num_context,input_dim_x)
            y_context (tensor): context point's y values (batch,num_context,input_dim_y)
        Returns:
            tensor: latent representation of the context (batch,1,latent_dim)
        """

        assert len(x_context.shape) == 3, \
            'Incorrect shapes: ensure x_context is a rank-3 tensor.'
        assert len(y_context.shape) == 3, \
            'Incorrect shapes: ensure y_context is a rank-3 tensor.'

        # representation r
        x = torch.cat((x_context, y_context), dim=-1)
        x = self.pre_pooling(x)
        r = torch.mean(x, dim=-2, keepdim=False)

        return r

class Classifier(nn.Module):
    """ Classifier mapping r to the probabilities for the different classes
    Args:
        classifier_layer_widths (list of int): list with teh dimensionality of the layers from the aggregate r to the classification output
    """

    def __init__(self, classifier_layer_widths):
        super(Classifier, self).__init__()
        h_classifier = nn.ModuleList([])
        l = len(classifier_layer_widths)
        for i in range(0, l - 1):
            h_classifier.append(nn.Linear(classifier_layer_widths[i], classifier_layer_widths[i + 1]))
            if i < l - 2:
                h_classifier.append(nn.ReLU())
        self.classifier = nn.Sequential(*h_classifier)
        self.classifier_activation = nn.Softmax(dim=-1)

    def forward(self,r):
        """ Forward pass through the classifier

        Args:
            r (tensor): representation of the context points (batch, dim_r)

        Returns:
            tensor: probabilities for the different classes (batch, num_classes)
        """

        logits = self.classifier(r)
        probs = self.classifier_activation(logits)
        return logits, probs


class LatentNetwork(nn.Module):
    """ Latent network mapping the aggregate r and the one-hot encoding of the class to the parameters of the distribution of the continuous latent variable

    Args:
         latent_network_layer_widths (list of int): list with teh dimensionality of the layers form the aggregate r to the continuous latent parameters
    """

    def __init__(self,latent_network_layer_widths):
        super(LatentNetwork, self).__init__()

        self.latent_dim = latent_network_layer_widths[-1]
        assert self.latent_dim % 2 == 0, "The dimension of the last layer of the latent network should be divisible by 2 (mean + std)"

        h_latent = nn.ModuleList([])
        l = len(latent_network_layer_widths)
        for i in range(0, l - 1):
            h_latent.append(nn.Linear(latent_network_layer_widths[i], latent_network_layer_widths[i + 1]))
            if i < l - 2:
                h_latent.append(nn.ReLU())
        self.latent_embedding = nn.Sequential(*h_latent)

    def forward(self,r_repeated,one_hot):
        """ Foward pass through the latent network giving the parametrisation of the distribution over the continuous latent variables

        Args:
            r_repeated (tensor): aggregate r, repeated over the batch dimension to match the shape of the one-hot encoding of the samples (batch * num_samples, dim_r)
            one_hot (tensor): one-hot encoding of the samples (batch * num_samples, num_classes)

        Returns:
            tensor: parametrization of the distribution of the continuous latent variable (batch * num_samples, 2 * latent_dim_size)
        """

        x = torch.cat((r_repeated,one_hot), dim = -1)
        parameters = self.latent_embedding(x)
        mean, log_std = torch.split(parameters,self.latent_dim//2,dim=-1)
        std = torch.exp(log_std)
        return mean, std

class Decoder(nn.Module):
    """Decoder used for the NP model.

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

    def forward(self, x_target, one_hot, z):
        """Forward pass through the decoder
        Args:
            x_target (tensor): target locations (batch * num_samples(, num_samples), num_target, input_dim_x)
            one_hot (tensor): class latent variable (batch * num_samples(, num_samples), num_classes)
            z (tensor): continuous latent variable (batch * num_samples(, num_samples), latent_dim)

        Returns:
            tensor: predicted mean at every target points (batch, num_target, output_dim)
        """

        # Reshape inputs to model.

        latent = torch.cat((one_hot,z),dim=-1)
        # Repeat the latent once for each input target points.
        if len(latent.shape) == 2:
            num_target = x_target.shape[1]
            latent = torch.unsqueeze(latent, 1)
            latent = latent.repeat(1, num_target, 1)
        elif len(latent.shape) == 3:
            num_target = x_target.shape[2]
            latent = torch.unsqueeze(latent, 2)
            latent = latent.repeat(1, 1, num_target, 1)
        elif len(latent.shape) == 4:
            num_target = x_target.shape[3]
            latent = torch.unsqueeze(latent, 3)
            latent = latent.repeat(1, 1, 1, num_target, 1)
        else:
            raise RuntimeError(f'Latent shape not supported "{latent.shape}".')

        x = torch.cat((x_target, latent), -1)
        x = self.post_pooling(x)

        return x

class GaussianSampler(nn.Module):
    """ Sampler to sample from a Gaussian posterior distribution with the reparametrization trick
    """

    def __init__(self):
        super(GaussianSampler, self).__init__()
        pass

    def forward(self, mean, std):
        """

        Args:
            mean (tensor): mean of the Gaussian distribution to sample from (batch_size,*dim)
            std (tensor): standard deviation of the Gaussian distribution to sample from (batch_size,*dim)

        Returns:
            tensor: sample from the Gaussian posterior distribution
        """
        epsilon = torch.randn(mean.size(), device=mean.device)

        return mean + std * epsilon


if __name__ == "__main__":
    encoder_layer_widths = [3, 128, 128, 128]
    decoder_layer_widths = [2, 128, 128, 128, 1]
    classifier_layer_widths = [128,128,10]
    latent_network_layer_widths = [138,128,128]
    prior = "UnitGaussian"
    model = NP(encoder_layer_widths, decoder_layer_widths, classifier_layer_widths, latent_network_layer_widths, prior)
    summary(model,[(10,2),(10,1),(100,2)])

    x_context = torch.rand((4,8,2))
    y_context = torch.rand((4,8,1))
    x_target = torch.rand((4,20,2))
    y_target = torch.rand((4, 20, 1))
    target_label = torch.round(torch.rand(4) * 9).int()
    target_label[:2] = -1
    num_samples = 10
    opt = torch.optim.Adam(model.parameters(),1e-4,weight_decay=1e-5)
    alpha = 0.1
    num_samples_expectation = 2
    parallel = True
    std_y = 0.1
    J = model.joint_train_step(x_context,y_context,x_target,target_label,y_target,opt,alpha,num_samples_expectation)
    J = model.joint_loss(x_context,y_context,x_target,target_label,y_target,alpha,num_samples_expectation=16,std_y=std_y,parallel=parallel)











