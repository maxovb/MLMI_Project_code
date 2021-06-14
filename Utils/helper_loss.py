import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

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

    component_weights = Categorical(weights)
    component_parameters = Normal(loc=mean,scale=std)
    dist = MixtureSameFamily(component_weights,component_parameters)
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