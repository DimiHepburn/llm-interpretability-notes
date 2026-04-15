"""llm-interpretability-notes: Tools and research notes on mechanistic interpretability."""
from .attention_analyser import (
    compute_attention_entropy,
    identify_specialised_heads,
    compute_head_importance,
    attention_distance_profile
)
