from dataclasses import dataclass
from torch import nn
import torch

# https://arxiv.org/abs/2002.05202v1
# Feed Fowrard Network from the papaer "Swish-Gated Linear Units for Neural Network Layers"

@dataclass
class Config:
    n_embd: int = 384
    vocab_size: int = 65
    seq_length: int = 256
    dropout: float = 0.2
    n_heads: int = 6
    n_layers: int = 6
    ffn_bias: bool = True

class GLU(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        self.l1 = nn.Linear(model_args.n_embd, model_args.n_embd * 4, bias = model_args.ffn_bias)
        self.l2 = nn.Linear(model_args.n_embd, model_args.n_embd * 4, bias = model_args.ffn_bias)
        self.l3 = nn.Linear(model_args.n_embd * 4, model_args.n_embd, bias = model_args.ffn_bias)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(model_args.dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        a = self.sigmoid(self.l1(x))
        b = self.l2(x)
        
        x = torch.mul(a , b)
        
        return self.dropout(self.l3(x))
