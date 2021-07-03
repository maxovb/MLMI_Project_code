import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.independent import Independent

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

    discr_logp = - nn.BCELoss()(probs_same_image, target_discr,reduction=reduction)

    # compute the accuracy
    predicted = (probs_same_image > 1 / 2).type(torch.float)
    total_discriminator = len(target_discr)
    if total_discriminator != 0:
        accuracy_discriminator = ((predicted == target_discr).sum()).item() / total_discriminator
    else:
        accuracy_discriminator = 0

    return discr_logp, accuracy_discriminator, total_discriminator

