

import torch
import os
os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset, DatasetDict
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, IntervalStrategy, GPTJForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
cache = "/scratch/project/dd-21-23/test/cache"

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache)
tokenizer.pad_token = tokenizer.eos_token

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

text = """Generate corresponding SQL.  Schema Json: {"stadium": ["stadium id", "location", "name", "capacity", "highest", "lowest", "average"], "singer": ["singer id", "name", "country", "song name", "song release year", "age", "is male"], "concert": ["concert id", "concert name", "theme", "stadium id", "year"], "singer in concert": ["concert id", "singer id"]} Question: What is the total number of singers who are above 23? SQl:  ### """
generated = tokenizer(text, return_tensors="pt").input_ids.cuda()
sample_outputs = model.generate(generated, do_sample=True, top_k=50,
                                max_length=128, top_p=0.95, temperature=0.9, num_return_sequences=20)
for i, sample_output in enumerate(sample_outputs):
    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
