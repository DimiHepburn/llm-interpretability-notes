# Key Papers in Mechanistic Interpretability

A curated reading list with annotations, organised by theme.

---

## Foundational Work

### Elhage et al. (2021) — A Mathematical Framework for Transformer Circuits
- **Core idea**: Attention heads and MLP layers can be understood as performing interpretable computations on residual stream vectors
- **Key insight**: The residual stream acts as a "communication channel" between layers — analogous to working memory in prefrontal cortex
- **Neuro-AI parallel**: Just as neuroscientists decompose neural circuits into functional motifs, this paper decomposes transformer computation into interpretable circuits

### Olah et al. (2020) — Zoom In: An Introduction to Circuits
- **Core idea**: Features in neural networks are built from interpretable sub-circuits
- **Key insight**: Polysemantic neurons (neurons that respond to multiple unrelated features) are a fundamental challenge — reminiscent of mixed selectivity in neuroscience
- **Neuro-AI parallel**: Mixed selectivity in prefrontal cortex neurons serves a computational purpose (high-dimensional representation); polysemanticity in transformers may serve a similar function

---

## Attention Head Specialisation

### Clark et al. (2019) — What Does BERT Look At?
- **Finding**: Different attention heads specialise for different linguistic relations (syntax, coreference, etc.)
- **Neuro-AI parallel**: This mirrors the functional specialisation observed in cortical columns — specific neural populations tuning to specific feature dimensions

### Voita et al. (2019) — Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting
- **Finding**: Only a small subset of attention heads are truly important; many can be pruned
- **Neuro-AI parallel**: This echoes findings from lesion studies — not all brain regions contribute equally to any given task. The concept of "neural necessity" applies to attention heads

---

## Superposition & Feature Splitting

### Elhage et al. (2022) — Toy Models of Superposition
- **Core idea**: Neural networks represent more features than they have dimensions, using superposition (compressed, overlapping representations)
- **Key insight**: This is a fundamental obstacle to interpretability — features are not cleanly separable
- **Neuro-AI parallel**: Population coding in neuroscience: information is distributed across many neurons, with each neuron participating in multiple representations. Superposition is the computational analog

---

## Activation Patching & Causal Tracing

### Meng et al. (2022) — Locating and Editing Factual Associations in GPT
- **Method**: Causal tracing via activation patching to identify which layers store factual knowledge
- **Neuro-AI parallel**: Directly analogous to optogenetic manipulation in neuroscience — activating/silencing specific components to establish causal (not just correlational) involvement

---

## Emerging Directions

- **Sparse autoencoders** for extracting monosemantic features from polysemantic neurons (Cunningham et al., 2023; Bricken et al., 2023)
- **Feature steering** — modifying model behaviour by intervening on specific features at inference time
- **Developmental interpretability** — studying how features emerge during training (analogous to developmental neuroscience)
