import torch


def nansum(x,dim=None):
    try:
        if dim == None :
            return torch.nansum(x)
        else:
            return torch.nansum(x,dim=dim)
    except AttributeError:
        x2 = x.clone()
        x2[x2 == float('nan')] = 0
        if dim == None :
            return torch.sum(x2)
        else:
            return torch.sum(x2,dim=dim)
    