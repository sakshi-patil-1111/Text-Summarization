from datasets import load_dataset
from transformers import AutoTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data, test_data = dataset["train"], dataset["test"]

# Tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing
def preprocess_function(examples):
    inputs = tokenizer(examples["article"], max_length=1024, truncation=True, padding="max_length")
    targets = tokenizer(examples["highlights"], max_length=256, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_train = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
tokenized_test = test_data.map(preprocess_function, batched=True, remove_columns=test_data.column_names)

# Use subset for faster training
tokenized_train = tokenized_train.shuffle(seed=42).select(range(10000))
tokenized_test = tokenized_test.shuffle(seed=42).select(range(2000))

# Load pre-trained model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Training Arguments — auto save model
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    evaluation_strategy="no",
    save_strategy="epoch",   # <== Save model every epoch
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="no",
    fp16=True,
    dataloader_num_workers=2,
    gradient_accumulation_steps=2  
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Train and save
trainer.train()
trainer.save_model("fine_tuned_model")  # ✅ Saves full model to use later
