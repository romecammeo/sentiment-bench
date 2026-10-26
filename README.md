# SentimentBench

**Customer-review sentiment classification: fine-tuning, evaluation, and error analysis**

A small hackathon project investigating how a pretrained sentiment classifier behaves before and after parameter-efficient (LoRA) fine-tuning on customer reviews. It also retrieves similar reviews and generates a short contextual reply in a minimal Streamlit application.

## Research question

Does LoRA fine-tuning improve sentiment-classification performance on held-out customer reviews, and which errors remain or change?

## Instructor-required deliverables

- [ ] Run a pretrained sentiment classifier and record baseline predictions.
- [ ] Fine-tune with LoRA using a separate training split.
- [ ] Evaluate both versions on **the same held-out reviews**.
- [ ] Report accuracy, precision, recall, and inference latency.
- [ ] Compare misclassified examples before and after fine-tuning.
- [ ] Retrieve similar reviews and generate a short reply using retrieved context.
- [ ] Build a small Streamlit page.
- [ ] Document how to reproduce the experiment and record a two-minute demo.

## Implementation sequence

1. **Baseline:** get a small dataset into Python, load a model, make predictions.
2. **Evaluation:** implement metrics and a before/after comparison format.
3. **Adaptation:** fine-tune with LoRA and evaluate on the unchanged held-out examples.
4. **Retrieval + reply:** retrieve similar reviews and construct a short context-aware response.
5. **UI + submission:** build Streamlit, check failure cases, document and demonstrate.

## Proposed layout

```
sentiment-bench/
  README.md
  .gitignore
  requirements.txt
  docs/experiment_protocol.md
  src/            # create modules as we implement them
  tests/          # add focused tests with implementation
  app.py          # Streamlit interface, later
```

Only the setup documents are supplied initially; the application code will be built incrementally during the hackathon.

## Reproducibility and responsible claims

Training, validation, and test data must be kept separate. The pretrained and fine-tuned classifiers will be scored on the **same untouched evaluation split**. Record the model and dataset identifiers, sample sizes, random seed, package versions, training settings, and hardware before reporting results. Do not claim that fine-tuning improves performance until the measured results support it.

The full protocol is in [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## Status

Repository preparation only. No trained model, evaluation results, or working app are claimed yet.
