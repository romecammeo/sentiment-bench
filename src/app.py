
import sys
from pathlib import Path

import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
)
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# Allow importing our existing source modules.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from retrieve import load_reviews, retrieve_similar, EMBEDDING_MODEL
from generate_reply import generate_reply, GENERATOR_NAME


MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
ADAPTER_PATH = ROOT / "models" / "sentiment-lora"


st.set_page_config(
    page_title="SentimentBench",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_resources():
    # Load both sentiment classifiers.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    baseline = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    adapted_base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    adapted = PeftModel.from_pretrained(
        adapted_base,
        str(ADAPTER_PATH),
    )

    baseline_classifier = pipeline(
        "text-classification",
        model=baseline,
        tokenizer=tokenizer,
        device=-1,
    )

    adapted_classifier = pipeline(
        "text-classification",
        model=adapted,
        tokenizer=tokenizer,
        device=-1,
    )

    # Load and embed the historical Sunboard review sample.
    reviews = load_reviews()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    review_embeddings = embedding_model.encode(
        [review["review_text"] for review in reviews],
        convert_to_tensor=True,
    )

    # Load the generative model.
    generator_tokenizer = AutoTokenizer.from_pretrained(
        GENERATOR_NAME
    )

    generator = AutoModelForCausalLM.from_pretrained(
        GENERATOR_NAME
    )

    device = (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    generator = generator.to(device)
    generator.eval()

    return (
        baseline_classifier,
        adapted_classifier,
        reviews,
        embedding_model,
        review_embeddings,
        generator_tokenizer,
        generator,
    )


st.title("SentimentBench")
st.caption(
    "Sentiment evaluation, semantic retrieval, "
    "and AI-assisted customer-response drafting."
)

st.info(
    "Demonstration project. Generated replies require human review."
)

review_text = st.text_area(
    "Customer review",
    value="The backrest wobbles when I lean backwards.",
    height=120,
)

if st.button("Analyze review", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        with st.spinner("Loading models and analyzing feedback..."):
            (
                baseline_classifier,
                adapted_classifier,
                reviews,
                embedding_model,
                review_embeddings,
                generator_tokenizer,
                generator,
            ) = load_resources()

            baseline_result = baseline_classifier(
                review_text,
                truncation=True,
                max_length=128,
            )[0]

            adapted_result = adapted_classifier(
                review_text,
                truncation=True,
                max_length=128,
            )[0]

            matches = retrieve_similar(
                review_text,
                reviews,
                embedding_model,
                review_embeddings,
                top_k=3,
            )

            reply = generate_reply(
                review_text,
                matches,
                generator_tokenizer,
                generator,
            )

        st.subheader("Sentiment comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pretrained classifier",
                baseline_result["label"],
            )
            st.write(
                f"Model score: {baseline_result['score']:.3f}"
            )

        with col2:
            st.metric(
                "LoRA-adapted classifier",
                adapted_result["label"],
            )
            st.write(
                f"Model score: {adapted_result['score']:.3f}"
            )

        st.caption(
            "Classification scores are not necessarily calibrated "
            "probabilities of correctness."
        )

        st.subheader("Similar historical reviews")

        for match in matches:
            with st.expander(
                f"{match['review_id']} — {match['title']} "
                f"(similarity: {match['similarity']:.3f})"
            ):
                st.write(match["review_text"])
                st.caption(
                    f"Variant: {match['variant']} | "
                    f"Rating: {match['stars']} stars"
                )

        st.subheader("Suggested customer-support reply")
        st.text_area(
            "Draft for human review",
            value=reply,
            height=160,
        )

        st.caption(
            "Historical reviews are unverified customer reports. "
            "Similarity does not establish factual agreement."
        )
