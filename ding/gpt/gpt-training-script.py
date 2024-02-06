from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
from tqdm import tqdm
import wandb

torch.set_printoptions(precision=4, linewidth=140, sci_mode=False)

with open('ding/gpt/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - seq_length, (batch_size,))
    x = torch.stack([data[i:i+seq_length] for i in ix])
    y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_length = 256 
batch_size = 64

@dataclass
class Config:
    n_embd: int = 384
    vocab_size: int = vocab_size
    seq_length: int = seq_length
    dropout: float = 0.2
    n_heads: int = 6
    n_layers: int = 6

xb,yb = get_batch('train')
print("x shape",xb.shape)
#-------------------------------------------


class GPT(nn.Module):
    def __init__(self, model_args: Config, custom_ffn=None):
        super().__init__()
        
        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.n_embd)
        self.position_emb = nn.Embedding(model_args.seq_length, model_args.n_embd)
        
        self.lm_head = nn.Linear(model_args.n_embd, model_args.vocab_size)
        self.blocks = nn.ModuleList([Block(model_args, custom_ffn) for _ in range(model_args.n_layers)])
        self.ln = nn.LayerNorm(model_args.n_embd)
        
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.tril(torch.ones(model_args.seq_length, model_args.seq_length, device=device)).view(
                1,1, model_args.seq_length, model_args.seq_length)
            self.register_buffer("mask", mask)
        else:
            self.mask = None
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        
    def forward(self, x, targets=None):
        B,T = x.shape

        tok_emb = self.token_emb(x) #  (B,T,C)
        pos_emb = self.position_emb(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        
        for layer in self.blocks:
            x = layer(x, self.mask)
        x= self.ln(x)
        x = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = x.shape
            x = x.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(x, targets)

        return x, loss
    
class SelfAttention(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_embd = model_args.n_embd
        self.dropout = model_args.dropout
        self.scale = (model_args.n_embd/model_args.n_heads) ** -0.5
        assert self.n_embd % self.n_heads == 0, "Embedding size must be 0 modulo number of heads."
        
        # Architecutre
        self.k = nn.Linear(model_args.n_embd, model_args.n_embd)
        self.q = nn.Linear(model_args.n_embd, model_args.n_embd)
        self.v = nn.Linear(model_args.n_embd, model_args.n_embd)
        self.proj = nn.Linear(model_args.n_embd, model_args.n_embd)

        self.norm = nn.LayerNorm(model_args.n_embd)
        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.resoults_dropout = nn.Dropout(model_args.dropout)

        # Flash Attention
        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")  
    
    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        B,T,C = x.shape
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Transpose the Tensor to get the number of heads
        q = q.view(B, T, self.n_heads, self.n_embd//self.n_heads)
        k = k.view(B, T, self.n_heads, self.n_embd//self.n_heads)
        v = v.view(B, T, self.n_heads, self.n_embd//self.n_heads)
        # Permute the Tensor again
        q = q.permute(0, 2, 1, 3) # B n_heads seq_length embd_size(per head)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3) 
       
        # flash_attn  // did improve training on XTX7900 from 1it/s to 8.5it/s
        if self.flash_attn:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                )
        else:
            s = q@k.transpose(2,3) / self.scale
            s = s.masked_fill(mask == 0, float('-inf'))
            out = s.softmax(dim=-1)@v
            self.attn_dropout(out)
        
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        
        out = self.proj(out)
        out = self.resoults_dropout(out)
        return out

class FeedFowardNetwork(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(model_args.n_embd, model_args.n_embd*4),
            nn.GELU(),
            nn.Linear(model_args.n_embd*4, model_args.n_embd),
            nn.Dropout(model_args.dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, model_args: Config, custom_ffn= None):
        super().__init__()

        self.attn = SelfAttention(model_args)
        self.ffn = custom_ffn(model_args) if custom_ffn else FeedFowardNetwork(model_args)
        self.ln1 = nn.LayerNorm(model_args.n_embd)
        self.ln2 = nn.LayerNorm(model_args.n_embd)
    
    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
        

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(Config)

m = model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Initialize wandb
wandb.init(project="Attenction-from-scratch", name="gpt")

max_iters = 5000
eval_interval = 100
eval_iters = 200


for iter in tqdm(range(max_iters), desc="Training Progress"):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"\nstep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        wandb.log({"step": iter, "train_loss": losses['train'], "val_loss": losses['val']})

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    x, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    wandb.log({"iteration_loss": loss.item()})

# Save the model
torch.save(model.state_dict(), "gpt.pth")