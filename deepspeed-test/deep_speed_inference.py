import torch
import os
import pandas as pd
import numpy as np

os.environ["WANDB_DISABLED"] = "true"
from transformers import AutoTokenizer, GPTJForCausalLM

# torch.backends.cuda.matmul.allow_tf32 = True
cache = "/scratch/project/dd-21-23/test/cache"

torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir=cache)
tokenizer.pad_token = tokenizer.eos_token

model = GPTJForCausalLM.from_pretrained("./results/checkpoint-750/", local_files_only=True).cpu()


def ask_client(query, dframe, max_length=100):
    column_names = list(data.columns.values)
    input = f"""Generate corresponding SQL. The table name is "insurance_data" Schema Json: "insurance_data":{column_names} Question: {query} SQL: """
    print(input + "\n \n")
    tokens = tokenizer(input, return_tensors="pt").input_ids.cpu()

    # for temperature_now in np.arange(0.1, 2.0, 0.1):
    sample_outputs = model.generate(tokens,
                                    do_sample=True,
                                    early_stopping=True,
                                    top_k=50,
                                    max_length=350,
                                    top_p=0.95,
                                    temperature=0.9,
                                    num_return_sequences=2)
    for i, sample_output in enumerate(sample_outputs):
        current_output = tokenizer.decode(sample_output, skip_special_tokens=True)
        output = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        before, sep, after = output.partition('SELECT')
        query = sep + after
        print("{}: {}\n".format(i, query))

    output = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    before, sep, after = output.partition('SELECT')
    query = sep + after
    print("Query: " + query + '\n')
    print("Result: \n")
    try:
        result = ps.sqldf(query, globals())
        return result
    except:
        print("Failed to execute query")
    # print(result)
    return "None"

DATA_CSV_FILE = './gistfile1.txt'

data = pd.read_csv(DATA_CSV_FILE, sep=';')
data.name = 'data'

your_question = "How many people are over 43 and have a Saab?"
question_2="Show me the customers with insured zip 466132"
ask_client(question_2, data)
