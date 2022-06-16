import wandb

wandb.init(project="gpt-j experiments", entity="vyomkesh")

import torch
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, IntervalStrategy, GPTJForCausalLM

cache = "/scratch/project/dd-21-23/test/cache"

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache)
tokenizer.pad_token = tokenizer.eos_token
training_args = TrainingArguments(output_dir='./results', num_train_epochs=1.5, logging_steps=1,
                                  save_strategy="steps",
                                  save_steps=100,
                                  save_total_limit=2,
                                  per_device_train_batch_size=4, per_device_eval_batch_size=4, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True,
                                  deepspeed='./ds_config_gpt_j.json', evaluation_strategy="epoch", eval_steps=100,
                                  report_to=["wandb"])

model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache).cuda()
# model.resize_token_embeddings(len(tokenizer))
train_dataset = load_dataset('text', data_files='./train.txt')
validation_dataset = load_dataset('text', data_files='./validation.txt')
raw_datasets = DatasetDict(
    {
        "train": train_dataset['train'].shuffle(),  # .shuffle().select(range(50000)),
        "validation": validation_dataset['train'].shuffle(),  # .shuffle().select(range(500))
    }
)

context_length = 128


class CodeGenDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(txt, truncation=True,
                                       max_length=max_length, padding='max_length')
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


dataset = CodeGenDataset(train_dataset['train']['text'], tokenizer, max_length=context_length)

# tokenized_datasets = raw_datasets.map(
#     tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
# )
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    'labels': torch.stack([f[0] for f in data])}).train()

generated = tokenizer("<|startoftext|>", return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                bos_token='<|startoftext|>',
                                eos_token='<|endoftext|>', pad_token='<|pad|>',
                                max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
