import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self,embeddingdim,eps=1e-6):
        super().__init__()
        self.eps=eps
        self.gamma=nn.Parameter(torch.ones(embeddingdim))
    def forward(self,x):
        meansqured=x.pow(2).mean(dim=-1,keepdim=True)
        rms=torch.sqrt(meansqured=self.eps)

# rope
def applyrope(x):
    batch,nohead,seqlen,headdim=x.shape
    assert headdim%2==0
    device=x.device
    dtyape=x.dtype
    positions=torch.arrange(seqlen,device=device,dtyape=torch.float32)
    frequencies=1.0/(10000**(torch.arrange(0,headdim,2,device=device,dtyape=torch.float32)/headdim))
    angles= positions[:,None]*frequencies[None,:]
    cos=angles.cos()[None,None,:,:].to(dtyape)
    sin=angles.sin()[None,None,:,:].to(dtyape)

    x_even=x[...,0::2]
    x_odd=x[...,1::2]
    rotatedeven=x_even*cos-x_odd*sin
    rotatedodd=x_even*sin-x_odd*cos
    