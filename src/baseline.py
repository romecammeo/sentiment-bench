from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

reviews = [
    "The headphones sound great, but the battery barely lasts an hour.",
    "Delivery took longer than expected, although the product itself is excellent.",
    "It's not as disappointing as I expected.",
    "The interface is simple, but I wouldn't recommend the subscription."
]

results = classifier(reviews)

for review, result in zip(reviews, results):
    print(review)
    print(result)
    print()



predictions = []

for result in results:
    predictions.append(result["label"])



print(predictions)
print(len(predictions))