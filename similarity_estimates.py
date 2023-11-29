import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

def tsm(x, use_sf=True):
    """ 
    summary
    
    Function for Temporal Self-similarity Matrix (TSM) from: https://openaccess.thecvf.com/content_CVPR_2020/papers/Dwibedi_Counting_Out_Time_Class_Agnostic_Video_Repetition_Counting_in_the_CVPR_2020_paper.pdf

    Args:
        x (torch.tensor): Tensor of takens/features of shape B x C x T to get the TSM for

    Returns:
        sim: tensor of shape B x T x T 
    """ 
    x = x.permute(0,2,1)# B x T x C
    sim = -torch.cdist(x,x) # B x T x T
    if use_sf:
        sim = F.softmax(sim,dim=-1) 
    return sim



def get_topk(x,kp,kn=None, return_idx=False, sort=False):
    """ 
    summary
    
    Function for using `tsm` to return the top k closest/furthest features over dimension T. Setting the return_idx to `True` will only return the indices.

    Args:
        x (torch.tensor): Tensor of takens/features of shape B x C x T 
        kp (int): Integer for the top-k closest (temporal) features to be returned.
        kn (int,optional): Integer for the top-k furthest (temporal) features to be returned. Defaults to None, which will result to be initialised as kn=kp.
        return_idx (bool, optional): Boolean flag to determine if only the indices are to be returned.
        sort (bool, optional): Boolean flag to determine if the temporal order is to be priororised over distance in the output ordering.

    Returns:
        if `return_idx` is `True`:
            sidx_p, idx_n (tuple): Tuple of tensors contaning the B x T x kp and B x T x kn closest/furthest. 
        else:
            px, pn (tuple): Tuple of tensors containing B x C x T x kp and B x C x T x kn closest/furthest features.
    """
    if not kn:
        kn = kp
    simmat = tsm(x, use_sf=False) # get sim matrix
    _,idx_p = torch.topk(simmat,k=kp,dim=-1) # B x T x kp
    _,idx_n = torch.topk(simmat,k=kn,dim=-1,largest=False) # B x T x kn
    if sort:
        idx_p,_ = torch.sort(idx_p,dim=-1)
        idx_n,_ = torch.sort(idx_n,dim=-1)
    
    if return_idx:
        return idx_p, idx_n
    
    idx_p = einops.repeat(idx_p, 'B T k -> B T C k',C=x.shape[1])
    idx_n = einops.repeat(idx_n, 'B T k -> B T C k',C=x.shape[1])
    
    px = torch.stack([x.gather(-1,idx_p[:,i,:,:].squeeze(1)) for i in range(idx_p.shape[1])],dim=-2).to(x.device)
    pn = torch.stack([x.gather(-1,idx_n[:,i,:,:].squeeze(1)) for i in range(idx_n.shape[1])],dim=-2).to(x.device)
    
    return px, pn



def mean_thres(x,beta, return_masks=False):
    """ 
    summary
    
    Function for using `tsm` to return the closest/furthest features over dimension T basedon the mean+/-beta. Setting the return_idx to `True` will only return the indices.
    ! NOTE : as the number of close/far pairs is not fixed instead of temporal indices, temporal 0-1 masking is used. The returned tensors are of size B x T x T x C with the thresholded locations being zeroed.

    Args:
        x (torch.tensor): Tensor of takens/features of shape B x C x T 
        beta (float): Float number to be used alongside the mean for thresholding.
        return_masks (bool, optional): Boolean flag to determine if only the masks are to be returned.

    Returns:
        if `return_idx` is `True`:
            mask_p, mask_n (tuple): Tuple of masks contaning the B x T x T and B x T x t closest/furthest. 
        else:
            px, pn, mask_p, mask_n (tuple): Tuple of tensors containing B x C x T x T and B x C x T x T closest/furthest features and B x T x T and B x T x t closest/furthest masks.
    """
    simmat = tsm(x, use_sf=False)
    simmat_m = torch.mean(simmat, dim=-1, keepdim=True)
    
    mask_p = F.threshold((simmat-simmat_m+beta) * -1,0,0) # keep the negatives when subtracting the mean and adding the beta
    mask_n = F.threshold(simmat-simmat_m-beta,0,0) # keep the positives when subtracting the mean and beta
    mask_p[mask_p > 0] = 1. # 1-0 threshold
    mask_n[mask_n > 0] = 1.
    
    if return_masks:
        return mask_p, mask_n
    
    x = einops.repeat(x, 'B C T -> B T R C',R=x.shape[-1])
    
    px = einops.rearrange(x * mask_p.unsqueeze(-1), 'B T1 T2 C -> B C T1 T2')
    pn = einops.rearrange(x * mask_n.unsqueeze(-1), 'B T1 T2 C -> B C T1 T2')
    
    return px, pn, mask_p, mask_n



def distance_relative(x,p):
    """ 
    summary
    
    Function for using `tsm` to return the closest/furthest features over dimension T based on p%.
    ! NOTE : as the number of close/far pairs is not fixed instead of temporal indices, temporal 0-1 masking is used. The returned tensors are of size B x T x T x C with the thresholded locations being zeroed.

    Args:
        x (torch.tensor): Tensor of takens/features of shape B x C x T 
        p (float): Float number (0.0 < p < 1.0) for the closest/furthest % of temporal locations to be returned.

    Returns:
        px, pn, mask_p, mask_n (tuple): Tuple of tensors containing B x C x T x T and B x C x T x T closest/furthest features and B x T x T and B x T x t closest/furthest masks.
    """
    simmat = tsm(x, use_sf=False) # all values are negative
    min_simmat,_ = torch.min(simmat,dim=-1, keepdim=True) # lowest similarity
    max_simmat,_ = torch.max(simmat,dim=-1, keepdim=True) # highest similarity
    
    thres = torch.abs(max_simmat - min_simmat) * p
    
    mask_p = torch.threshold(simmat + thres,0,0)
    mask_n = torch.threshold(-simmat + thres,0,0)
    
    x = einops.repeat(x, 'B C T -> B T R C',R=x.shape[-1])
    
    px = einops.rearrange(x * mask_p.unsqueeze(-1), 'B T1 T2 C -> B C T1 T2')
    pn = einops.rearrange(x * mask_n.unsqueeze(-1), 'B T1 T2 C -> B C T1 T2')
    
    return px, pn, mask_p, mask_n


## testing
if __name__=='__main__':
    
    x = torch.randn(64,512,16)
    px,pn = get_topk(x,kp=3)
    print(px.shape,pn.shape)
    px,pn,_,_ = mean_thres(x,beta=1e-12)
    print(px.shape,pn.shape)
    px,pn,_,_ = distance_relative(x,p=0.1)
    print(px.shape,pn.shape)
    
    
    