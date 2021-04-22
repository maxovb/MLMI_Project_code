import torch
from torch.distributions.normal import Normal

def gaussian_logpdf(target, mean, std, reduction=None):
    """Gaussian log-density. (copied from https://github.com/cambridge-mlg/convcnp.git)
    Args:
        target (tensor): Inputs.
        mean (tensor): Mean.
        std (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".
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
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')