from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Config
import torch
from torch import nn
import os

# Set the WANDB_PROJECT environment variable
os.environ['WANDB_PROJECT'] = 'lora'

config = GPT2Config.from_pretrained('gpt2-medium', attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium', config=config).to('cuda')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to('cuda')

r = 4
alpha = 32
k,d = model.transformer.h[0].attn.c_attn.weight.shape
## Beggining of the LORA CODE

## CREATE
B = torch.zeros((d,r))# torch.zero like // d x r
A = torch.normal(mean=0, std=1, size=(r, k)) # N(0,sigma^2) # torch rand like // r x k
print(B@A)
# iterate over the layers and freeze the weights
for param in model.parameters():
    param.requires_grad = False



class Lora(nn.Module):
    def __init__(self, r, alpha):
        super(Lora, self).__init__()
        self.r = r
        self.aplha = alpha


class LoraGPT2(Lora, nn.Module):
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
        x_1 = self.model(x)
        x_2 = x @ self.A @ self.B # ABx
        x=  x_1 + x_2 * self.scale
        return x

class LoraLinear(Lora, nn.Module):
    pass


# i divide r //2 since we take kvq and projection matrix
w = []
for i in range(r//2):
    model.transformer.h[23-i].attn.c_attn = LoraGPT2(model.transformer.h[23-i].attn.c_attn, r, alpha)
    model.transformer.h[23-i].attn.c_proj = LoraGPT2(model.transformer.h[23-i].attn.c_proj, r, alpha)
    w.append([model.transformer.h[23-i].attn.c_attn.A, model.transformer.h[23-i].attn.c_attn.B, model.transformer.h[23-i].attn.c_proj.A, model.transformer.h[23-i].attn.c_proj.B])

## TIME TO TRAIN

# load in the data for E2E NLG Challenge
with open('ding/lora/data/train.txt', encoding='utf-8') as f:
    train = f.read().split(' \n')
with open('ding/lora/data/valid.txt', encoding='utf-8') as f:
    valid = f.read().split(' \n')
tokenizer.pad_token = tokenizer.eos_token 
train_encodings = tokenizer(train, truncation=True, padding=True, max_length=512,return_tensors='pt')
val_encodings = tokenizer(valid, truncation=True, padding=True, max_length=512,return_tensors='pt')

class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Ensure that 'labels' are included and set to the input_ids for language modeling tasks
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()  # Duplicate input_ids to labels for autoregressive language modeling
        return item



train_dataset = GPT2Dataset(train_encodings)
val_dataset = GPT2Dataset(val_encodings)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    learning_rate=2e-4,
    lr_scheduler_type='linear',
    evaluation_strategy="epoch",
    label_smoothing_factor=0.1,
    report_to="wandb",
    run_name="lora-gpt-2_finetune",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model('ding/lora/models/')
