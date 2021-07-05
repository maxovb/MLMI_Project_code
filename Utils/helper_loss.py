import torch
import random
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.independent import Independent
from torch.distributions.kl import kl_divergence

def gaussian_logpdf(target, mean, std, reduction=None, start_idx_sum=1):
    """Gaussian log-density. (copied from https://github.com/cambridge-mlg/convcnp.git)
    Args:
        target (tensor): Inputs.
        mean (tensor): Mean.
        std (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", "batched_mean" and "samples_mean".
        start_idx_sum (int, optional): if reduction is samples_mean, start summing from that index
    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=std)
    logp = dist.log_prob(target)

    # number of dimensions
    num_dim = len(logp.shape)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, list(range(1,num_dim))))
    elif reduction == 'samples_mean':
        if start_idx_sum < 0:
            idx = len(logp.shape) + start_idx_sum
        else:
            idx = start_idx_sum
        return torch.mean(torch.sum(logp, list(range(idx, num_dim))),dim=idx-1)
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

def mixture_of_gaussian_logpdf(target, mean, std, weights, reduction=None, start_idx_sum=1):
    """Gaussian log-density. (copied from https://github.com/cambridge-mlg/convcnp.git)
    Args:
        target (tensor): Inputs (batch, *).
        mean (tensor): Mean. (batch, num_components, *)
        std (tensor): Standard deviation. (batch, num_components, *)
        weights (tensor): Component weights (batch, num_components, *)
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", "batched_mean".
    Returns:
        tensor: Log-density.
    """

    component_weights = Categorical(weights.type(torch.float))
    component_parameters = Normal(loc=mean,scale=std)
    event_dim = len(mean.shape) - 2
    component_parameters_multivariate = Independent(component_parameters,event_dim)

    dist = MixtureSameFamily(component_weights,component_parameters_multivariate)
    logp = dist.log_prob(target)

    # number of dimensions
    num_dim = len(logp.shape)

    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, list(range(1,num_dim))))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

def discriminator_logp(probs_same_image,reduction="mean"):
    batch_size = probs_same_image.shape[0]

    assert batch_size == 1 or batch_size % 2 == 0, "The batch size should be 1 (if only one batch example), or a multiple of 2"

    if batch_size > 2:
        target_discr = torch.ones(batch_size).to(probs_same_image.device)
        target_discr[batch_size // 2:] = 0
    else:
        target_discr = torch.ones(1).to(probs_same_image.device)

    discr_logp = - nn.BCELoss(reduction=reduction)(probs_same_image, target_discr)

    # compute the accuracy
    predicted = (probs_same_image > 1 / 2).type(torch.float)
    total_discriminator = len(target_discr)
    if total_discriminator != 0:
        accuracy_discriminator = ((predicted == target_discr).sum()).item() / total_discriminator
    else:
        accuracy_discriminator = 0

    return discr_logp, accuracy_discriminator, total_discriminator

def consistency_loss(output_logit, num_sets_of_context=1,reduction="mean"):

    batch_size = output_logit.shape[0]

    # obtain the probability distribution
    probs = nn.Softmax(dim=-1)(output_logit)

    assert num_sets_of_context == 2, "Consistency loss does not handle other number of context sets than 2 at the moment"

    # get the original batch size
    single_set_batch_size = batch_size / num_sets_of_context
    assert single_set_batch_size == int(single_set_batch_size), "The tensor batch size should be a multiple of the number of sets of context (when using consistency regularization), but got batch size: " + str(mean.shape[0]) + " and num of context sets: " + str(num_sets_of_context)
    single_set_batch_size = int(single_set_batch_size)

    # split between the two sets of context sets
    probs_set1, probs_set2 = torch.split(probs, single_set_batch_size, dim=0)

    # avoid numerical issues:
    probs_set1 = torch.clamp(probs_set1,1e-3)
    probs_set2 = torch.clamp(probs_set2, 1e-3)
    probs_set1 = probs_set1/torch.sum(probs_set1,dim=-1,keepdim=True) # re-normalize
    probs_set2 = probs_set2 / torch.sum(probs_set2,dim=-1,keepdim=True) # re-normalize


    loss = js_divergence(probs_set1,probs_set2, reduction=reduction)

    if batch_size > 2:

        assert batch_size % 2 == 0, "The batch size should be divisible by two, repeat every image twice with two context sets"

        indices = torch.ones(batch_size//2,device=probs.device)
        for i in range(batch_size//2):
            while True:
                r = random.randint(0,batch_size//2-1)
                if r != i: # check that we don't compare two same images
                    break
            indices[i] = r

        probs_compare = probs_set2[indices.type(torch.int64)]
        loss += - js_divergence(probs_set1, probs_compare, reduction=reduction)

    return loss


def js_divergence(probs_set1, probs_set2, reduction="mean"):
    """Jenson-Shannon divergence between the two probabilties distributions
    """
    # compute the Jensen Shannon divergence
    m = (probs_set1 + probs_set2)/2
    loss = 0.0
    dist1 = Categorical(probs_set1)
    dist2 = Categorical(probs_set2)
    distm = Categorical(m)
    loss += kl_divergence(dist1,distm)
    loss += kl_divergence(dist2,distm)

    if reduction == "mean":
        div = 0.5 * torch.mean(loss)
    elif reduction == "sum":
        div = 0.5 * torch.sum(loss)
    else:
        raise NotImplementedError("reduction for JS divergence is implemented only with mean (default) and sum")
    return div

if __name__ == "__main__":
    x = torch.tensor([[2.,0,0],[0,2.,0],[0,0,2.],[2.,0,0],[0,2.,0],[0,0,2.]],requires_grad=True).type(torch.float)
    l = consistency_loss(x,num_sets_of_context=2)

