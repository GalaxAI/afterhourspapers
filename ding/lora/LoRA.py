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
                 fan_in_fan_out:bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
                 **kwargs):
        
        nn.Linear.__init__(self, in_f, out_f,**kwargs)
        Lora.__init__(self, lora_dropout,merge_weights,r,lora_alpha)

        self.A = nn.Parameter(self.weight.new_zeros((r,in_f)))
        self.B = nn.Parameter(self.weight.new_zeros((out_f,r)))
        self.scale = r/lora_alpha


        self.fan_in_fan_out = fan_in_fan_out
        # We are reseting the parameters for linear since we load in the weights later
        nn.Linear.reset_parameters(self)
        self.weight.requires_grad = False
        nn.init.kaiming_uniform_(self.A)
        nn.init.zeros_(self.B)
        if self.fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)


    def forward(self, x):
            def T(w):
                return w.transpose(0,1) if self.fan_in_fan_out else w
            x = self.lora_dropout(x)
    
            # Calculate the adapted weights
            adapted_weights = self.weight + T(self.B @ self.A * self.scale)
            
            # Perform the linear operation with adapted weights
            return nn.functional.linear(x, adapted_weights, self.bias)

class LoraEmbedding(Lora, nn.Module):
    pass

# Can extend to the other layers the idea is the same.
