
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from retrieve import (
    load_reviews,
    retrieve_similar,
    EMBEDDING_MODEL,
)


GENERATOR_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def generate_reply(query, matches, tokenizer, generator):
    """Generate a customer-support draft using retrieved reviews."""

    # Build a short context from the retrieved reviews.
    context = "\n".join(
        f"- Review {match['review_id']}: {match['review_text'][:220]}"
        for match in matches
    )

    # Give the model its instructions and the information to use.
    messages = [
        {
            "role": "system",
            "content": (
                "You draft polite, concise customer-support replies. "
                "Treat previous customer reviews as unverified experiences. "
                "Different reviewers may report contradictory experiences. "
                "Do not invent product guarantees, repairs, or refund policies. "
                "Do not claim an unstable product is safe. "
                "Write a draft that a human should review before sending."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Customer complaint:\n{query}\n\n"
                f"Previous customer reviews:\n{context}\n\n"
                "Write a short reply of two or three sentences. "
                "Acknowledge the customer's concern, avoid unsupported "
                "claims, and suggest contacting customer support."
            ),
        },
    ]

    # Convert chat messages into numerical model inputs.
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(generator.device)

    # Generate new tokens without calculating training gradients.
    with torch.inference_mode():
        output_ids = generator.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove the original prompt tokens from the generated sequence.
    input_length = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_length:]

    # Convert generated token IDs back into readable text.
    reply = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return reply.strip()


def main():
    # Load historical Sunboard reviews.
    reviews = load_reviews()

    # Prepare semantic retrieval.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    review_texts = [
        review["review_text"]
        for review in reviews
    ]

    review_embeddings = embedding_model.encode(
        review_texts,
        convert_to_tensor=True,
    )

    # Example customer complaint.
    query = "The backrest wobbles when I lean backwards."

    # Find the three most similar historical reviews.
    matches = retrieve_similar(
        query,
        reviews,
        embedding_model,
        review_embeddings,
        top_k=3,
    )

    # Load the text-generation model and its tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)

    generator = AutoModelForCausalLM.from_pretrained(
        GENERATOR_NAME
    )

    # Apple Silicon GPU when available; otherwise CPU.
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    generator = generator.to(device)
    generator.eval()

    # Generate a reply using the retrieved reviews.
    reply = generate_reply(
        query,
        matches,
        tokenizer,
        generator,
    )

    # Display the results.
    print("\nCustomer complaint:")
    print(query)

    print("\nRetrieved reviews:")
    for match in matches:
        print(
            f"- {match['review_id']} "
            f"(similarity: {match['similarity']:.3f}): "
            f"{match['title']}"
        )

    print("\nGenerated reply:")
    print(reply)


if __name__ == "__main__":
    main()
