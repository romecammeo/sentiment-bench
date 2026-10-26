# Submission and review plan

**Hackathon deadline:** Wednesday, 28 October 2026, **09:00** (course local time).

## State after Monday, 26 October

The developer has demonstrated locally:
- DistilBERT baseline inference and Yelp evaluation;
- a completed PEFT LoRA run and a 200-review before/after comparison;
- semantic retrieval over a private historical Sunboard review sample;
- a Qwen-based customer-support draft;
- an integrated Streamlit interface showing baseline and adapted sentiment, retrieved reviews, and draft reply.

**Repository synchronization gap:** at the time this checklist was written, only `src/baseline.py` and starter documents were visible on the remote development branch. The other Python files must be committed and pushed from the Mac. This checklist does not imply those files are already public.

## Intended source layout

```text
README.md
requirements.txt
docs/
  experiment_protocol.md
  results_2026-10-26.md
  submission_plan.md
src/
  baseline.py
  evaluate.py
  train_lora.py
  compare_models.py
  sunboard_data.py
  retrieve.py
  generate_reply.py
  app.py
data/raw/                       # Private; Git-ignored
models/sentiment-lora/          # Generated locally; Git-ignored
```

Streamlit is presently invoked using `python -m streamlit run src/app.py` from the repository root. Avoid documenting a root-level `app.py` until the file is actually moved.

## Before Wednesday submission

- [ ] Inspect local `git status --short` for intended code files and unintended private data / generated outputs.
- [ ] Commit and push the source files on `feature/baseline-evaluation`.
- [ ] Update `requirements.txt` with all actually used dependencies (including `torchvision` if required by this environment) and record tested versions; don't blindly claim portability from one Mac.
- [ ] Keep private Sunboard review text, personal information, `.venv`, secrets, and model checkpoints out of Git.
- [ ] Provide a legally shareable, clearly labelled synthetic or otherwise permitted sample dataset so other users can run the retrieval demo without the private CSV; document where the private dataset belongs locally.
- [ ] Document how to regenerate or otherwise lawfully obtain the LoRA adapter (model files are Git-ignored). Avoid claiming a fresh clone runs without setup.
- [ ] Verify the Streamlit app from the repository root; test empty input and at least one mixed-sentiment customer review.
- [ ] Preserve the actual evaluation procedure, performance metrics, and failure limitations in the README.
- [ ] Prepare and record a **two-minute demo** showing the two classifiers, retrieved evidence, generated draft, and one limitation.
- [ ] Confirm instructor collaborator `zivuch` has accepted the invitation and complete the course submission requirements.

## Proposed demo narrative (rough timing)

| Time | Content |
| --- | --- |
| 0:00–0:20 | Problem and real-world historical review case study |
| 0:20–0:45 | Pretrained versus LoRA evaluation; accuracy improved 87.5% to 90.0% on 200 reviews |
| 0:45–1:20 | Live Streamlit sentiment predictions and three retrieved similar reviews |
| 1:20–1:45 | Generated draft reply and why it requires human review |
| 1:45–2:00 | Limits: small sample, slight negative-recall regression, retrieval contradictions |

## Tuesday's learning priorities

Reconstruct the complete data flow without reading the implementation line-by-line. Focus on:
- **Why** the train, validation, and evaluation examples have different jobs;
- what the tokenizer, model, head, LoRA adapters, embedding model, similarity search, and generator each do;
- how precision, recall, confusion matrices, and latency derive from actual comparisons;
- how the saved adapter differs from original base-model weights;
- how code, private data, generated model artifacts, and public reproducible instructions should be separated.

Do not expand the architecture before ensuring reproducibility, submission assets, and code understanding.
