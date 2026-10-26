import csv
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "sunboard_reviews_anonymized.csv"
)

with DATA_PATH.open(encoding="utf-8-sig", newline="") as file:
    reviews = list(csv.DictReader(file))

print("Reviews loaded:", len(reviews))
print("First review ID:", reviews[0]["review_id"])
print("First review text:", reviews[0]["review_text"])