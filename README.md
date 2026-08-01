# Smart MCQ Solver Challenge — DL & GenAI Project

Kaggle competition solution for academic multiple-choice question answering, comparing a fine-tuned transformer against two models written from scratch.

**Student:** Ashrita Khanta ; **Roll No:** 22f1000828
**Course:** BSDA2001P — Introduction to DL and GenAI Project

---

## Problem

Each row has a question prompt and five options (A–E). The model must predict the correct option. The metric is **MAP@3**, so the submission ranks its top three guesses instead of giving one hard answer.

Because of MAP@3, the random baseline is `(1 + 1/2 + 1/3) / 5 = 0.3667`, not 0.20 — anything below that is worse than guessing.

## Main design decision

I treated this as an **option scoring** problem, not a five-class classification problem.

```
question + option A ─┐
question + option B ─┤
question + option C ─┼─► same encoder ─► shared Linear(→1) ─► 5 scores ─► softmax ─► CrossEntropyLoss
question + option D ─┤
question + option E ─┘
```

A five-class setup never reads the options at all — "A" is just a label that means something different in every row. Option scoring instead learns question–option compatibility. The single **shared** scoring head (not five separate heads) is what makes the option-shuffling augmentation valid, and stacking the five scores before softmax makes the options compete with each other.

All three models use this identical interface, so the only thing that changes between them is the encoder. That is what makes the architecture comparison fair.

## Models

| # | Model | Params | Notes |
|---|-------|--------|-------|
| 1 | DeBERTa v3 Base (fine-tuned) | ~184M | `AutoModelForMultipleChoice`, disentangled attention, fp16, grad accumulation |
| 2 | TextCNN (custom) | ~0.2M | Kernels [2,3,4] × 100 filters, masked max-pool over time |
| 3 | Bi-LSTM (from scratch) | ~0.2M | Hand-written LSTM cell, no `nn.LSTM`, fused gates, forget-gate bias = 1.0 |

Three logged configurations per model — **9 runs total**, all tracked in Weights & Biases.

## Results

| Model | Val MAP@3 | Key observation |
|-------|-----------|-----------------|
| Bi-LSTM (from scratch) | **1.0000** | Perfect in all 3 configs, slowest to converge (epoch 9) |
| TextCNN | **1.0000** | Perfect in all 3 configs, fastest overall |
| DeBERTa v3 (frozen emb.) | 0.9200 | Best transformer run |
| DeBERTa v3 (lr 1e-5) | 0.5711 | Learning, not converged in 3 epochs |
| DeBERTa v3 (lr 2e-5) | 0.4039 | Collapsed — loss stuck at ln(5) |
| Ensemble of 3 DeBERTa runs | 0.9017 | Worse than best single model, rejected |
| Longest-option heuristic | 34.95% acc | Strongest baseline with no training |

**Kaggle leaderboard: 0.75810 MAP@3 — rank 164.**

## The interesting part: 1.0000 → 0.7581

The perfect validation score was **not** a good result — it was a warning. EDA had already flagged why:

- 2,000 rows, 2,980 unique words, **0.00% OOV rate** — a small closed vocabulary
- Very templated question phrasing
- **242 duplicate prompts** that leak across a random split

So the validation set contained questions the model had already memorised, while the leaderboard did not. The 0.24 drop is the cost of that leakage. Six runs tied at 1.0000 also meant the validation set could no longer rank my own models.

Honest conclusion: not "CNNs beat transformers", but **on a small, closed-vocabulary, templated dataset, a 0.2M-parameter model matches a 184M-parameter one — capacity should match the data, not be maximised.**

## Most useful single finding

Freezing DeBERTa's embedding layer took validation MAP@3 from **0.4039 → 0.9200** at the same learning rate.

The embedding matrix is 128,000 × 768 ≈ 98M parameters — roughly half the model. Updating that with 1,700 training examples is mostly noise. Freezing it lets the optimiser spend its budget on the transformer layers and the new head instead.

## EDA highlights

| Check | Finding | Why it mattered |
|-------|---------|-----------------|
| Answer distribution | B = 24.5% (vs 20% uniform) | Motivated option-order shuffling |
| Prompt length | mean 18.15 words, 95% < 31 | Set `max_len = 128` for word-level models |
| Longest option | 34.95% accuracy | Strong no-training baseline; length artefact exists |
| Trick questions (NOT/EXCEPT) | 0 found | Removes the main argument against a CNN |
| OOV rate | **0.00%** | Subword tokenization buys nothing here |
| Duplicate prompts | 242 in train | Source of the validation leakage |

## Tokenization

| Property | Custom SimpleTokenizer | DeBERTa v3 (SentencePiece) |
|---|---|---|
| Granularity | Whole words | Subword |
| Vocab used | 2,982 | 128,000 (pretrained) |
| OOV on test | 0.00% | 0.00% |
| Sequence length | 128, fixed padding | 256 cap, dynamic padding (~63 typical) |
| Separator | literal string | real `[SEP]` token |

Dynamic padding via a custom `collate_fn` cut a typical batch from 256 to ~63 tokens — about 4× less compute with no information lost.

## Repo contents

```
notebooks/    Kaggle notebooks for all three models
report/       Final project report (PDF)
figures/      Architecture diagrams and W&B curves
submission.csv
```

## Reproducing

Seed 42, 85/15 stratified split (1,700 train / 300 validation). Each notebook runs top-to-bottom on a Kaggle GPU and writes `submission.csv`.

## What I would do differently

- **GroupKFold by prompt** so the 242 duplicates cannot land on both sides of a split — this is the main fix
- **A fair budget for DeBERTa**: 8–10 epochs with early stopping, layer-wise LR decay, a proper LR sweep
- **A leakage ablation**: drop the duplicates and retrain everything
- **A harder test set**: rephrase held-out questions to test meaning vs. word overlap
- **A real `[SEP]` token** plus segment embeddings in the from-scratch tokenizer
- **Weighted ensembling** instead of an equal-weight average dragged down by two weak runs

## Links

- **Weights & Biases (all 9 runs):** https://wandb.ai/22f1000828-t1-2026-indian-institute-of-technology-madras/DL-22f1000828-notebook-t22026
- **Kaggle — Model 1 (DeBERTa v3):** https://kaggle.com/code/ashritakhanta/dl-22f1000828-notebook-t22026?scriptVersionId=337886436
- **Kaggle — Model 2 (TextCNN):** https://kaggle.com/code/ashritakhanta/dl-22f1000828-notebook-t22026?scriptVersionId=337998076
- **Kaggle — Model 3 (Bi-LSTM):** https://kaggle.com/code/ashritakhanta/dl-22f1000828-notebook-t22026?scriptVersionId=337870064

## References

1. He, P., Gao, J., Chen, W. (2021). *DeBERTaV3.* arXiv:2111.09543
2. He, P., Liu, X., Gao, J., Chen, W. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* arXiv:2006.03654
3. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014
4. Hochreiter, S., Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation 9(8)
5. Loshchilov, I., Hutter, F. (2017). *Decoupled Weight Decay Regularization (AdamW).* arXiv:1711.05101
6. Gers, F. A., Schmidhuber, J., Cummins, F. (2000). *Learning to Forget.* Neural Computation 12(10)
7. Wolf, T., et al. (2020). *Transformers: State-of-the-Art NLP.* EMNLP 2020
8. Paszke, A., et al. (2019). *PyTorch.* NeurIPS 2019
9. Biewald, L. (2020). *Experiment Tracking with Weights and Biases.*
