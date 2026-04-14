# 🔍 LLM Interpretability Notes

> *Opening the black box: understanding what large language models actually learn, represent, and do internally.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-Mechanistic%20Interp-9B59B6?style=flat-square)](https://github.com/neelnanda-io/TransformerLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Overview

This repository is a living collection of **research notes, paper summaries, experiment logs, and code explorations** on mechanistic interpretability of large language models.

Mechanistic interpretability asks a simple but profound question: *what computations are transformer models actually performing?* Not what they do at a behavioural level, but what the actual circuits, representations, and algorithms are inside the weights.

This matters enormously for three reasons:
1. **Safety** — we cannot reliably control systems we do not understand
2. **Alignment** — understanding representations helps us verify whether models have learned the right things
3. **Science** — LLMs are the largest and most complex learning systems ever built; understanding them advances our understanding of learning and intelligence itself

---

## 📓 Notes Index

| # | Topic | Key Concepts | Status |
|---|-------|--------------|--------|
| 01 | [Transformer Architecture Review](notes/01_transformer_review.md) | Attention, MLP layers, residual stream | ✅ Complete |
| 02 | [The Residual Stream as a Communication Channel](notes/02_residual_stream.md) | Information flow, layer contributions | ✅ Complete |
| 03 | [Attention Heads: What Do They Actually Attend To?](notes/03_attention_heads.md) | Induction heads, copy suppression, query composition | ✅ Complete |
| 04 | [Superposition and Polysemanticity](notes/04_superposition.md) | Feature geometry, interference, sparse coding | 🔄 In Progress |
| 05 | [Sparse Autoencoders for Feature Extraction](notes/05_sparse_autoencoders.md) | SAEs, monosemanticity, feature decomposition | 🔄 In Progress |
| 06 | [Circuit Analysis: Indirect Object Identification](notes/06_ioi_circuit.md) | Wang et al. 2022, causal interventions | 📋 Planned |
| 07 | [Activation Patching & Causal Tracing](notes/07_activation_patching.md) | ROME, causal mediation analysis | 📋 Planned |
| 08 | [Knowledge Localisation in Transformers](notes/08_knowledge_localisation.md) | Factual associations, MLP as key-value memory | 📋 Planned |

---

## 🧪 Key Experiments

### Experiment 1: Induction Head Detection

Induction heads are one of the most well-documented circuit elements in transformer models. They implement a simple algorithm: if the sequence contains [...A B ... A], predict B. They emerge in two-layer attention-only models and are implicated in in-context learning.

```python
import transformer_lens
from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained("gpt2")

def detect_induction_heads(model, seq_len=50):
    """
    Detect induction heads by their characteristic pattern:
    high attention score on token [pos - (seq_len-1)] for each position.
    """
    # Create a repeated random sequence: [A B C ... A B C ...]
    random_tokens = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
    repeated = torch.cat([random_tokens, random_tokens], dim=1)

    # Run with cache to capture attention patterns
    _, cache = model.run_with_cache(repeated)

    induction_scores = {}
    for layer in range(model.cfg.n_layers):
        attn_pattern = cache["pattern", layer]  # [batch, head, dest, src]
        # Induction score: mean attention to position [i - seq_len + 1]
        score = attn_pattern[0, :, seq_len:, 1:seq_len].diagonal(dim1=-2, dim2=-1).mean(-1)
        induction_scores[layer] = score.detach()

    return induction_scores

scores = detect_induction_heads(model)
for layer, head_scores in scores.items():
    top_heads = head_scores.topk(3).indices.tolist()
    print(f"Layer {layer}: top induction head candidates: {top_heads}")
```

---

### Experiment 2: Probing Classifiers for Emotional Content

Do LLMs develop internal representations of sentiment and emotion, even when not explicitly trained for it? This experiment trains linear probes on intermediate activations to detect emotional content — and uses this to understand where emotional information is processed.

```python
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

def extract_activations(model, texts, layer_idx, hook_point="hook_resid_post"):
    """Extract residual stream activations at a given layer."""
    activations = []

    def hook_fn(value, hook):
        # Take the last token's activation as sentence representation
        activations.append(value[:, -1, :].detach().cpu().numpy())

    tokens_list = [model.to_tokens(t) for t in texts]
    for tokens in tokens_list:
        with model.hooks([(f"blocks.{layer_idx}.{hook_point}", hook_fn)]):
            model(tokens)

    return np.vstack(activations)

# Extract activations at each layer and train probes
results = {}
for layer in range(model.cfg.n_layers):
    train_acts = extract_activations(model, train_texts, layer)
    test_acts = extract_activations(model, test_texts, layer)

    probe = LogisticRegression(max_iter=1000)
    probe.fit(train_acts, train_labels)
    acc = probe.score(test_acts, test_labels)
    results[layer] = acc
    print(f"Layer {layer}: emotion probe accuracy = {acc:.3f}")
```

*Finding: Emotional information is distributed across layers, with a clear peak in middle layers — consistent with the hypothesis that semantic content is built up gradually through the residual stream.*

---

### Experiment 3: Attention Head Ablation Study

Systematically ablate attention heads and measure the effect on downstream tasks to identify which heads are causally important for specific capabilities.

```python
def ablation_study(model, task_tokens, task_labels, metric_fn):
    """
    Ablate each attention head and measure performance drop.
    Returns importance scores for each (layer, head) pair.
    """
    baseline_score = metric_fn(model, task_tokens, task_labels)
    importance = {}

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            # Zero out this head's output
            def zero_ablation_hook(value, hook, head_idx=head):
                value[:, :, head_idx, :] = 0
                return value

            hook_name = f"blocks.{layer}.attn.hook_z"
            with model.hooks([(hook_name, zero_ablation_hook)]):
                ablated_score = metric_fn(model, task_tokens, task_labels)

            importance[(layer, head)] = baseline_score - ablated_score

    return importance
```

---

## 📖 Paper Summaries

### Anthropic — "Towards Monosemanticity" (2023)
**Core claim**: Individual neurons in neural networks often respond to multiple unrelated concepts (polysemanticity), making them hard to interpret. Using **sparse autoencoders** trained on MLP activations, we can decompose the activation space into monosemantic features.

**Why it matters**: Opens a tractable path towards understanding what concepts a model has learned, beyond the tangled multi-dimensional structure of neurons.

---

### Wang et al. — "Interpretability in the Wild" (2022)
**Core claim**: The Indirect Object Identification (IOI) circuit in GPT-2 — which handles sentences like "When Mary and John went to the store, John gave a drink to ___" — can be fully reverse-engineered as a 26-component circuit.

**Why it matters**: Demonstrates that circuit-level understanding of real capabilities in production models is achievable, not just in toy models.

---

### Meng et al. — "ROME: Locating and Editing Factual Associations" (2022)
**Core claim**: Factual knowledge in GPT models is localised in specific MLP layers (mid-layer MLPs act as key-value memories), and can be surgically edited using **Rank-One Model Editing**.

**Why it matters**: If knowledge is localised and editable, we can correct factual errors, remove harmful beliefs, and understand how models store information.

---

### Elhage et al. — "A Mathematical Framework for Transformer Circuits" (2021)
**Core claim**: Provides a rigorous mathematical framework for analysing transformer computations, introducing the residual stream as a communication channel and decomposing attention as a composition of QK and OV circuits.

**Why it matters**: The foundational theoretical framework for mechanistic interpretability. Everything else builds on this.

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| [TransformerLens](https://github.com/neelnanda-io/TransformerLens) | Main interpretability toolkit — hooks, caching, patching |
| [baukit](https://github.com/davidbau/baukit) | Neural network dissection and intervention tools |
| [CircuitsVis](https://github.com/alan-cooney/CircuitsVis) | Visualise attention patterns and circuits |
| [sparse_autoencoder](https://github.com/ai-safety-foundation/sparse_autoencoder) | SAE training for feature extraction |

---

## 📂 Repository Structure

```
llm-interpretability-notes/
├── notes/
│   ├── 01_transformer_review.md
│   ├── 02_residual_stream.md
│   ├── 03_attention_heads.md
│   └── 04_superposition.md
├── experiments/
│   ├── 01_induction_heads.ipynb
│   ├── 02_emotion_probing.ipynb
│   └── 03_ablation_study.ipynb
├── paper_summaries/
│   ├── towards_monosemanticity.md
│   ├── ioi_circuit.md
│   ├── rome.md
│   └── mathematical_framework.md
├── src/
│   ├── probing.py
│   ├── patching.py
│   └── circuit_analysis.py
└── README.md
```

---

## 📚 Essential Reading

- Elhage et al. (2021). A Mathematical Framework for Transformer Circuits
- Wang et al. (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2
- Meng et al. (2022). Locating and Editing Factual Associations in GPT
- Anthropic (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning
- Lieberum et al. (2023). Does Circuit Analysis Interpretability Scale?

---

*Part of a broader research programme on neuroscience-inspired AI and AI humanisation.*
*See also: [neuro-ai-bridge](https://github.com/DimiHepburn/neuro-ai-bridge) | [humanising-ai](https://github.com/DimiHepburn/humanising-ai)*
