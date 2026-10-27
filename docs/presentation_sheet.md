# SentimentBench — presentation and articulation sheet

**For Tuesday, 27 October 2026.** Rehearsal aid for the two-minute hackathon demo and technical questions. Claims are based on the working local prototype and the reported October 26 experiment, not on production deployment.

## 1. One-sentence project description

> SentimentBench compares a pretrained sentiment classifier against a LoRA-adapted version on held-out Yelp reviews, then demonstrates semantic review retrieval and a human-reviewed customer-support draft using a historical product-review case study.

## 2. Explain the architecture in your own words

```text
             LABELLED YELP REVIEWS
                    |
         Train / validation / test
                    |
       LoRA adaptation of DistilBERT
                    |
   Evaluate original and adapted models
   on exactly the same held-out reviews
                    |
 Accuracy / precision / recall / latency
              + error changes

             USER REVIEW IN WEB UI
                    |
        +-----------+--------------+
        |                          |
  DistilBERT sentiment       MiniLM embeddings
  (original and adapted)     -> similar reviews
        |                          |
  Two classifications        retrieved context
                                   |
                              Qwen generator
                                   |
                         human-reviewed draft
```

The Yelp experiment and the Sunboard demonstration are related by the **business problem** but serve different methodological purposes.

## 3. Two-minute demo script (adapt the wording to your own speech)

**0:00–0:20 — The problem**

“My project is called SentimentBench. I wanted to examine not just whether an AI sentiment classifier works, but how its errors change after fine-tuning, and how that might matter in customer feedback. I also used a historical sample from a product I previously sold as the inspiration for the retrieval demonstration.”

**0:20–0:45 — The experiment**

“I started with a pretrained DistilBERT sentiment classifier. I adapted it with LoRA using 512 labelled Yelp training reviews, and I evaluated both versions on the same 200 held-out reviews. Accuracy went from 87.5% to 90%, but the trade-off matters: negative precision improved, while recall for negative reviews dropped slightly.”

**0:45–1:20 — Live demo**

“Here, I enter a customer complaint about a wobbling backrest. The application shows both sentiment predictions. Separately, an embedding model retrieves similar historical-style reviews. The public example data is synthetic; my local experiment also uses a historical sample of actual reviews.”

**1:20–1:45 — The generated reply**

“Those retrieved reviews are passed to a separate Qwen language model, which drafts a customer-support response. The draft is for human review; the sentiment classifier itself does not generate text.”

**1:45–2:00 — Limitations and take-away**

“One limitation I discovered is that semantic retrieval can return contradictory reviews. So a system can run successfully without producing trustworthy evidence. My main learning was to evaluate models and inspect failures, not just connect AI components.”

## 4. Articulation prompts — reconstruct without reading the script

**Q: What is the difference between the base model, LoRA adapters, and classification head?**

Expected distinction: the existing DistilBERT weights mostly remain frozen; LoRA adds trainable small matrices inside selected attention layers; the classification head maps text representations into sentiment class scores and is also trainable in this setup. The adapted model is loaded by combining the base checkpoint with saved adapter/head parameters.

**Q: Training vs. validation vs. test?**

Training examples update trainable parameters. Validation examples inform training choices without being used for gradient updates. Held-out test examples compare the original and adapted models using the same inputs, after model-selection decisions.

**Q: What do the main metrics mean?**

Accuracy = correct predictions / all predictions. When `NEGATIVE` is the class of interest: precision = correctly flagged negative reviews / all reviews flagged negative; recall = correctly flagged negative reviews / all genuinely negative reviews. Accuracy can rise while negative recall falls.

**Q: What is an embedding and what does cosine similarity prove?**

An embedding is a numerical vector representing text. Cosine similarity helps rank nearby representations; it does not prove that the two reviews agree, are factual, or are useful for drafting.

**Q: Why do we use three models?**

DistilBERT classifies sentiment; MiniLM embeds and retrieves similar reviews; Qwen generates text. Different architectures and objectives, not interchangeable functions.

**Q: Why do we keep the private CSV and model weights out of GitHub?**

Public code should be reproducible without distributing third-party review text or generated checkpoints. The public synthetic sample exercises retrieval, and the documented training script can regenerate an adapter.

**Q: What does Streamlit actually do?**

It is a Python-based user interface that invokes the loaded model functions, displays results, and reruns on interactions. Cached model resources prevent expensive reloads on every click.

## 5. Caveats to say accurately

- **19 reviews** refers to the particular **May historical log sample**, not the total number of reviews for the product.
- The reported metrics come from **Yelp test reviews**, not from the Sunboard review sample.
- Improvement was **87.5% to 90.0% on this 200-example comparison**, not universally.
- Fine-tuning made **6 incorrect predictions correct and 1 previously correct prediction incorrect**.
- Negative-review recall declined **89.9% to 88.9%** despite higher accuracy.
- The generated reply is coherent but generic; meaningful dependence on retrieved records was not demonstrated.
- The product remains a **prototype** and requires a human to review responses.

## 6. Filming sequence

1. Launch Streamlit before starting the recording; wait until the models are loaded if possible.
2. Keep the model-comparison table or notes visible briefly to substantiate the experiment.
3. Use the synthetic demonstration corpus for a shareable recording and identify it honestly.
4. Enter or show the wobbling-backrest review; click **Analyze review**.
5. Expand a retrieved result and show the generated draft.
6. Finish with one evidence-based limitation; keep the final video **at most two minutes**.

Stop the recording and inspect audio, screen legibility, and total duration. Save and submit the video and GitHub repository using the bootcamp's required submission workflow. Don't assume a running local Streamlit server is a public deployment.
