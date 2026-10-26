from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Load a small, reproducible sample from the test split
dataset = load_dataset(
    "fancyzhx/yelp_polarity",
    split="test"
)

dataset = dataset.shuffle(seed=42).select(range(100))

print("Dataset size:", len(dataset))
print("First example:", dataset[0])

# Convert dataset labels into model-compatible strings.
# Yelp Polarity uses 0 = negative, 1 = positive.
label_names = {0: "NEGATIVE", 1: "POSITIVE"}

reviews = list(dataset["text"])

actual = [label_names[label] for label in dataset["label"]]

classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
)

results = classifier(
    reviews,
    truncation=True,
    batch_size=8,
)

predicted = [result["label"] for result in results]

print("\nAccuracy:", accuracy_score(actual, predicted))
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