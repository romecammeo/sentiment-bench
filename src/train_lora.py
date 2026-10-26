from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 1. Training and validation reviews — NOT our test reviews
dataset = load_dataset("fancyzhx/yelp_polarity", split="train")
dataset = dataset.shuffle(seed=42).select(range(640))

train_data = dataset.select(range(512))
validation_data = dataset.select(range(512, 640))

# 2. Load the tokenizer and existing classifier
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def tokenize(batch):
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    encoded["labels"] = batch["label"]
    return encoded

train_data = train_data.map(
    tokenize, batched=True, remove_columns=train_data.column_names
)
validation_data = validation_data.map(
    tokenize, batched=True, remove_columns=validation_data.column_names
)

# 3. Attach small trainable LoRA adapters
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    modules_to_save=["pre_classifier", "classifier"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Configure training
training_args = TrainingArguments(
    output_dir="outputs/lora-run",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="none",
    dataloader_pin_memory=False,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    processing_class=tokenizer,
)

# 5. Train and save the adaptation
trainer.train()
print(trainer.evaluate())

model.save_pretrained("models/sentiment-lora")
tokenizer.save_pretrained("models/sentiment-lora")