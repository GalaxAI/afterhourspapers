from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, Trainer, TrainingArguments, GPT2Config, DataCollator
from datasets import load_dataset
from sacrebleu.metrics import BLEU
import torch
from torch import nn
import os

# Set the WANDB_PROJECT environment variable
os.environ['WANDB_PROJECT'] = 'lora'

config = GPT2Config.from_pretrained('gpt2-medium', attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium', config=config).to('cuda')


r = 4
alpha = 32

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
## TIME TO TRAIN

train,valid = load_dataset("e2e_nlg",split=['train','validation'])
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

def preprocess_and_tokenize(a):
    # Combine the input and output texts
    combined_texts = a['meaning_representation'] + '||' + a['human_reference']
    # Tokenize the combined texts
    return tokenizer(combined_texts, max_length=128, truncation=True, padding="max_length", return_tensors="pt")

# Apply preprocessing and tokenization to the train and validation datasets
tokenized_train = train.map(preprocess_and_tokenize)
tokenized_valid = valid.map(preprocess_and_tokenize)

# Assuming `tokenized_datasets` is your dataset prepared for training
tokenized_train = tokenized_train.map(lambda examples: {'labels': examples['input_ids']}, batched=True)
# Assuming `tokenized_datasets` is your dataset prepared for training
tokenized_valid = tokenized_valid.map(lambda examples: {'labels': examples['input_ids']}, batched=True)


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
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
)

trainer.train()

trainer.save_model('ding/lora/models/lora_v2')
trainer.evaluate()
