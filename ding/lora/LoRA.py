import torch
from torch import nn

class Lora(nn.Module):
    def __init__(self, r, alpha):
        super(Lora, self).__init__()
        self.r = r
        self.aplha = alpha
        # you can add more things to lora such as dropout, merge weights.


class LoraGPT2(Lora, nn.Module):
    """
    GPT2 is using Conv1D layer for attenction weights,
    but the calculations are the same as for linear
    """
    def __init__(self, model,r,alpha,**kwargs):
        nn.Module.__init__(self, **kwargs)
        Lora.__init__(self, r, alpha)
        self.model = model
        k,d = model.weight.shape
        device = model.weight.device
        self.A = nn.Parameter(self.model.weight.new_zeros((r,k)).transpose(0,1))
        self.B = nn.Parameter(self.model.weight.new_zeros((d,r)).transpose(0,1))
        self.scale = r/alpha
        self.model.weight.requires_grad = False

        ### Reseting the parameters
        nn.Linear.reset_parameters(self.model)
        # this is different than what is described in the paper but should not affect performance
        nn.init.kaiming_uniform_(self.A)
        nn.init.zeros_(self.B)
    
    def forward(self, x):
        x = self.model(x)
        x_2 = x @ self.A @ self.B * self.scale # ABx
        x += x_2 
        return x

class LoraEmbedding(Lora, nn.Module):
    pass

# Can extend to the other layers the idea is the same.
