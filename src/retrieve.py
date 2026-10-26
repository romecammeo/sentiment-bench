import csv
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "sunboard_reviews_anonymized.csv"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_reviews():
    with DATA_PATH.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def retrieve_similar(query, reviews, model, review_embeddings, top_k=3):
    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
    )

    matches = util.semantic_search(
        query_embedding,
        review_embeddings,
        top_k=top_k,
    )[0]

    return [
        {
            **reviews[match["corpus_id"]],
            "similarity": float(match["score"]),
        }
        for match in matches
    ]


def main():
    reviews = load_reviews()

    model = SentenceTransformer(EMBEDDING_MODEL)

    review_texts = [review["review_text"] for review in reviews]

    review_embeddings = model.encode(
        review_texts,
        convert_to_tensor=True,
    )

    query = "The backrest wobbles when I lean backwards."

    matches = retrieve_similar(
        query,
        reviews,
        model,
        review_embeddings,
    )

    print("Query:", query)

    for match in matches:
        print("\nReview ID:", match["review_id"])
        print("Title:", match["title"])
        print("Similarity:", round(match["similarity"], 3))
        print("Text:", match["review_text"][:400])


if __name__ == "__main__":
    main()