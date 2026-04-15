"""
Attention Pattern Analyser
==========================
Tools for visualising and analysing attention patterns in transformer models,
with a neuroscience-informed lens.

The key analogy: transformer attention mechanisms share structural similarities
with selective attention in biological neural systems — both involve competitive
selection, gating of information flow, and context-dependent weighting.

This module provides utilities for:
    1. Extracting attention matrices from transformer layers
    2. Computing attention entropy (a measure of attention "sharpness")
    3. Identifying attention heads that specialise (analogous to feature-selective neurons)

Author: Dimitri Romanov
Project: llm-interpretability-notes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_attention_entropy(attention_weights: np.ndarray) -> np.ndarray:
    """
    Compute Shannon entropy of attention distributions.
    
    High entropy → diffuse attention (analogous to distributed cortical processing)
    Low entropy → focused attention (analogous to spotlight attention / V1 selectivity)
    
    In neuroscience terms, this maps onto the distinction between:
    - Feature-based attention (broad, categorical)
    - Spatial attention (narrow, location-specific)
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention matrix of shape (n_heads, seq_len, seq_len)
    
    Returns
    -------
    np.ndarray
        Entropy per head per query position, shape (n_heads, seq_len)
    """
    # Clip to avoid log(0)
    attn = np.clip(attention_weights, 1e-10, 1.0)
    entropy = -np.sum(attn * np.log2(attn), axis=-1)
    return entropy


def identify_specialised_heads(
    attention_weights: np.ndarray,
    entropy_threshold: float = 1.5
) -> Dict[str, List[int]]:
    """
    Classify attention heads by their specialisation pattern.
    
    Analogous to how neuroscientists classify cortical neurons:
    - "Sharp" heads (low entropy) → like orientation-selective neurons in V1
    - "Broad" heads (high entropy) → like broadly-tuned neurons in higher cortical areas
    - "Positional" heads → like place cells encoding spatial position
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Shape (n_heads, seq_len, seq_len)
    entropy_threshold : float
        Threshold for classifying heads as sharp vs broad
    
    Returns
    -------
    dict
        Classification of each head
    """
    entropy = compute_attention_entropy(attention_weights)
    mean_entropy = np.mean(entropy, axis=-1)  # Average across positions
    
    classifications = {
        'sharp': [],      # Low entropy — focused attention
        'broad': [],      # High entropy — distributed attention
        'positional': [], # Attends primarily to nearby positions
    }
    
    n_heads, seq_len, _ = attention_weights.shape
    
    for h in range(n_heads):
        if mean_entropy[h] < entropy_threshold:
            classifications['sharp'].append(h)
        else:
            classifications['broad'].append(h)
        
        # Check for positional bias (diagonal dominance)
        diag_mass = np.mean([
            attention_weights[h, i, max(0, i-2):min(seq_len, i+3)].sum()
            for i in range(seq_len)
        ])
        if diag_mass > 0.5:
            classifications['positional'].append(h)
    
    return classifications


def compute_head_importance(
    attention_weights: np.ndarray,
    method: str = 'gradient'
) -> np.ndarray:
    """
    Estimate relative importance of each attention head.
    
    This connects to the neuroscience concept of "neural necessity" —
    just as lesion studies reveal which brain regions are necessary for
    a task, pruning attention heads reveals which are critical for
    model performance.
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Shape (n_heads, seq_len, seq_len)
    method : str
        'entropy' — heads with lower entropy are more specialised (potentially more important)
        'gradient' — placeholder for gradient-based importance (requires model access)
    
    Returns
    -------
    np.ndarray
        Importance score per head
    """
    if method == 'entropy':
        entropy = compute_attention_entropy(attention_weights)
        mean_entropy = np.mean(entropy, axis=-1)
        # Inverse entropy as proxy for importance (more focused = more important)
        max_entropy = np.log2(attention_weights.shape[-1])
        importance = 1.0 - (mean_entropy / max_entropy)
        return importance
    else:
        raise NotImplementedError(
            f"Method '{method}' requires model access. "
            f"Use 'entropy' for attention-only analysis."
        )


def attention_distance_profile(attention_weights: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute the attention distance profile for each head.
    
    Measures how far each head "looks" — analogous to receptive field
    sizes in visual cortex:
    - V1 neurons: small receptive fields (local attention)
    - V4/IT neurons: large receptive fields (global attention)
    
    Returns
    -------
    dict with:
        'mean_distance': average attention distance per head
        'std_distance': variability of attention distance per head
    """
    n_heads, seq_len, _ = attention_weights.shape
    
    # Distance matrix
    positions = np.arange(seq_len)
    dist_matrix = np.abs(positions[:, None] - positions[None, :])
    
    mean_distances = np.zeros(n_heads)
    std_distances = np.zeros(n_heads)
    
    for h in range(n_heads):
        weighted_dist = attention_weights[h] * dist_matrix
        mean_distances[h] = np.mean(np.sum(weighted_dist, axis=-1))
        std_distances[h] = np.std(np.sum(weighted_dist, axis=-1))
    
    return {
        'mean_distance': mean_distances,
        'std_distance': std_distances
    }


if __name__ == "__main__":
    print("=" * 60)
    print("LLM Interpretability: Attention Pattern Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate attention patterns for 8 heads, sequence length 20
    n_heads, seq_len = 8, 20
    
    # Create diverse synthetic attention patterns
    attn = np.random.dirichlet(np.ones(seq_len) * 0.5, size=(n_heads, seq_len))
    
    # Make head 0 very focused (sharp)
    attn[0] = np.eye(seq_len) * 0.8 + np.random.dirichlet(np.ones(seq_len) * 0.1, size=seq_len) * 0.2
    attn[0] /= attn[0].sum(axis=-1, keepdims=True)
    
    # Make head 1 positional (local attention)
    for i in range(seq_len):
        attn[1, i] = np.exp(-0.5 * ((np.arange(seq_len) - i) / 2)**2)
    attn[1] /= attn[1].sum(axis=-1, keepdims=True)
    
    # Analyse
    entropy = compute_attention_entropy(attn)
    classifications = identify_specialised_heads(attn)
    importance = compute_head_importance(attn, method='entropy')
    distances = attention_distance_profile(attn)
    
    print(f"\nHead classifications:")
    for category, heads in classifications.items():
        print(f"  {category}: {heads}")
    
    print(f"\nHead importance scores:")
    for h in range(n_heads):
        print(f"  Head {h}: importance={importance[h]:.3f}, "
              f"mean_distance={distances['mean_distance'][h]:.1f}")
    
    print(f"\nMost important head: {np.argmax(importance)}")
    print(f"Longest-range head: {np.argmax(distances['mean_distance'])}")
