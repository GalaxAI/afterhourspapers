from transformers import GPT2Tokenizer, GPT2Model,GPT2LMHeadModel
import torch
from torch import nn

import warnings
warnings.filterwarnings('ignore')


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to('cuda')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to('cuda')



r = 8
alpha = 8
k,d = model.transformer.h[0].attn.c_attn.weight.shape
## Beggining of the LORA CODE

## CREATE
B = torch.zeros((d,r))# torch.zero like // d x r
A = torch.normal(mean=0, std=1, size=(r, k)) # N(0,sigma^2) # torch rand like // r x k
print(B@A)
# iterate over the layers and freeze the weights
for param in model.parameters():
    param.requires_grad = False



class Lora(torch.nn.Module):
    def __init__(self, model, r, alpha):
        super(Lora, self).__init__()
        
        self.model = model
        self.model.weight.requires_grad = True


class LoraGPT2(Lora, nn.Module):
    def __init__(self, model, r, alpha,**kwargs):
        nn.Module.__init__(self, **kwargs)
        Lora.__init__(self, model, r, alpha)
        k,d = model.weight.shape
        device = model.weight.device
        self.A = torch.normal(mean=0,std=1,size=(r, k), device = device).transpose(0,1)
        self.B = torch.zeros((d, r),device = device).transpose(0,1)
        self.scale = r/alpha
    
    def forward(self, x):
        # x is the input
        x_1 = self.model(x)
        x_2 = x @ self.A @ self.B
        return x



# i divide r //2 since we take kvq and projection matrix
for i in range(r//2):
    model.transformer.h[23-i].attn.c_attn = Lora(model.transformer.h[23-i].attn.c_attn, r, alpha)
    model.transformer.h[23-i].attn.c_proj = Lora(model.transformer.h[23-i].attn.c_proj, r, alpha)

with torch.no_grad():
    output = model.generate(encoded_input.input_ids, max_length=50, attention_mask = encoded_input.attention_mask)  # Specify max_length as needed


# Decode the generated text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
