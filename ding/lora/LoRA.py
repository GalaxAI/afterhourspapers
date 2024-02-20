import torch
from torch import nn

class Lora():
    def __init__(self, lora_dropout:float,merge_weights:bool = False
                 ,r:int = 1, alpha:int = 1,):
        super(Lora, self).__init__()
        self.r = r
        self.aplha = alpha
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else: self.lora_dropout = lambda x: x # returns x
        
        self.merged = False
        self.merge_weights = merge_weights
    

class LoraLinear(nn.Linear,Lora):
    """
    GPT2 is using Conv1D layer for attenction weights,
    but the calculations are the same as for linear
    """
    def __init__(self,
                 in_f:int,
                 out_f:int,
                 r:int = 1,
                 lora_alpha:int = 1,
                 lora_dropout:float = 0.0,
                 merge_weights:bool = True,
                 **kwargs):
        
        nn.Linear.__init__(self, in_f, out_f,**kwargs)
        Lora.__init__(self, lora_dropout,merge_weights,r,lora_alpha)

        self.A = nn.Parameter(self.weight.new_zeros((r,in_f)))
        self.B = nn.Parameter(self.weight.new_zeros((out_f,r)))
        self.scale = r/lora_alpha

        # We are reseting the parameters for linear since we load in the weights later
        nn.Linear.reset_parameters(self)
        nn.init.kaiming_uniform_(self.A)
        nn.init.zeros_(self.B)
    
    def forward(self, x):
        x = self.weight + self.B @ self.A
        return nn.functional.linear(x, self.weight, self.bias)   

class LoraEmbedding(Lora, nn.Module):
    pass

# Can extend to the other layers the idea is the same.
