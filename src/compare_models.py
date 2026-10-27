import time

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from peft import PeftModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ADAPTER_PATH = "models/sentiment-lora"

# A fresh held-out subset, not used for training
dataset = load_dataset("fancyzhx/yelp_polarity", split="test")
dataset = dataset.shuffle(seed=42).select(range(100, 300))

reviews = list(dataset["text"])

label_map = {0: "NEGATIVE", 1: "POSITIVE"}
actual = [label_map[label] for label in dataset["label"]]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def evaluate_model(model, name):
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # Use CPU for a consistent initial comparison
    )

    # Warm up the model before timing
    classifier(reviews[:2], truncation=True, max_length=128)

    start = time.perf_counter()

    results = classifier(
        reviews,
        truncation=True,
        max_length=128,
        batch_size=8,
    )

    elapsed = time.perf_counter() - start

    predicted = [result["label"] for result in results]

    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(actual, predicted))
    print(
        "Negative precision:",
        precision_score(actual, predicted, pos_label="NEGATIVE"),
    )
    print(
        "Negative recall:",
        recall_score(actual, predicted, pos_label="NEGATIVE"),
    )
    print(
        "Confusion matrix:",
        confusion_matrix(
            actual,
            predicted,
            labels=["NEGATIVE", "POSITIVE"],
        ),
    )
    print("Average latency (ms/review):", round(elapsed / len(reviews) * 1000, 2))

    return predicted


# Original pretrained classifier
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

baseline_predictions = evaluate_model(
    base_model,
    "PRETRAINED BASELINE",
)

# Same original architecture + saved LoRA adaptation
adapted_base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

adapted_model = PeftModel.from_pretrained(
    adapted_base,
    ADAPTER_PATH,
)

adapted_predictions = evaluate_model(
    adapted_model,
    "LORA FINE-TUNED",
)


# Compare classification errors review by review
fixed = []
regressed = []

for i in range(len(reviews)):
    before_correct = baseline_predictions[i] == actual[i]
    after_correct = adapted_predictions[i] == actual[i]

    if not before_correct and after_correct:
        fixed.append(i)

    if before_correct and not after_correct:
        regressed.append(i)

print("\nErrors fixed by LoRA:", len(fixed))
print("New errors introduced by LoRA:", len(regressed))

for i in fixed[:3]:
    print("\nExample corrected by LoRA:")
    print("Review:", reviews[i][:300])
    print("Actual:", actual[i])
    print("Before:", baseline_predictions[i])
    print("After:", adapted_predictions[i])