import os
os.environ['HTTP_PROXY'] = "http://edcguest:edcguest@172.31.102.29:3128"
os.environ['HTTPS_PROXY'] = "http://edcguest:edcguest@172.31.102.29:3128"

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data, test_data = dataset["train"], dataset["test"]

print("Sample Article:\n", train_data[0]["article"])
print("Sample Summary:\n", train_data[0]["highlights"])

from transformers import AutoTokenizer

model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = tokenizer(examples["article"], max_length=1024, truncation=True, padding="max_length")
    targets = tokenizer(examples["highlights"], max_length=256, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

print("Tokenized Input:", tokenized_train[0]["input_ids"][:10])
print("Tokenized Label:", tokenized_train[0]["labels"][:10])

import pickle

with open("tokenized_train.pkl", "wb") as f:
    pickle.dump(tokenized_train, f)
with open("tokenized_test.pkl", "wb") as f:
    pickle.dump(tokenized_test, f)

tokenizer.save_pretrained("./saved_tokenizer")
