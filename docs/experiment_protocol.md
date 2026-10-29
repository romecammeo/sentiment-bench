# SentimentBench — initial experiment protocol

This document preserves the initial evaluation plan. See the [experiment record](results_2026-10-26.md) for the implemented configuration, measured results, and limitations.

## 1. Question and scope

**Question:** How does adapting a pretrained binary sentiment classifier to customer reviews affect accuracy, precision, recall, latency, and the kinds of examples it misclassifies?

Use a small, labelled binary-sentiment review dataset. A single existing classifier and one LoRA adaptation are enough for a first complete comparison. The exact model, corpus, subset size, hardware, and training method must be chosen and recorded during implementation.

## 2. Dataset separation

- **Training:** update LoRA/trainable parameters using labelled reviews.
- **Validation:** guide training choices (learning rate, epochs, checkpoint selection).
- **Final evaluation:** evaluate both the pretrained baseline and fine-tuned model on the *same untouched examples*.
- Keep near-duplicate reviews from leaking across splits when practical.
- Do not select hyperparameters using the final evaluation set. If it influences development decisions, document that limitation rather than calling the result an untouched test.

## 3. Baseline-versus-adapted comparison

For each evaluation review, store at least:

| review_id | true_label | baseline_prediction | adapted_prediction | baseline_latency_ms | adapted_latency_ms |
| --- | --- | --- | --- | --- | --- |
| (to populate) | | | | | |

Calculate accuracy, precision, and recall using the **same label definitions** for both versions. Record class balance and which class is considered positive. F1 and a confusion matrix are helpful additions if time permits.

Measure inference latency using the same environment and comparable measurement procedure, and state whether model loading is included. Use multiple examples or runs rather than interpreting a single timing as a stable benchmark.

## 4. Error analysis

Compare cases where:

- Both versions are correct.
- Both versions are wrong.
- Fine-tuning fixes an original error.
- Fine-tuning introduces a new error.

Optional linguistic hypotheses to inspect: negation, contrast, mixed or aspect-specific sentiment, and ambiguous wording. Record uncertainty and avoid assuming these categories have uniquely correct labels.

## 5. Retrieval and reply component

On user input, retrieve similar reviews (for example, using sentence embeddings and cosine similarity on a modest reference corpus). Display which reviews were retrieved. Use a distinct text-generation component to draft a short reply grounded in the retrieved material. Sentiment classification, similarity search, and generation are separate operations and should not be treated as interchangeable.

## 6. Reproducibility fields specified in the initial plan

- Dataset and license/source:
- Number of train / validation / evaluation examples:
- Random seed and splitting method:
- Baseline model and revision:
- LoRA configuration (targets, rank, alpha, dropout, trainable head):
- Learning rate, batch size, epochs:
- Hardware (CPU / GPU / Apple Silicon):
- Python and package versions:
- Evaluation code / command:
- Results and limitations:

