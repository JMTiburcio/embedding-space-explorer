"""
Experiment 03 — Phrase Embeddings

Demonstrates that embeddings can be generated from any string, not just
single words, and explores how context changes a vector.

The key distinction:
  - Token embedding  (embedding_utils): fixed lookup per token, no context.
  - Text embedding   (this file): full transformer pass over the whole string.
    Each token's vector is influenced by surrounding tokens via attention.

Experiments:
  1. Negation — does "not happy" land near "sad"?
  2. Phrase modifiers — does "a small dog" shift towards "puppy"?
  3. 3D map — positive/negative/neutral phrases in the same space.

Run:
  python 03_phrase_embeddings.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from embedding_utils import tokenizer, text_encoder, device


def get_text_embedding(text: str) -> torch.Tensor:
    """
    Full-sequence embedding via CLIP text encoder. Returns the [EOS] token
    vector [768], which accumulates context from the entire input string.

    Why [EOS]? The transformer's final token has attended to all previous
    tokens and acts as a summary representation — analogous to [CLS] in BERT.
    """
    inputs = tokenizer(
        text, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = text_encoder(inputs.input_ids.to(device))[0]  # [1, 77, 768]
    eos_idx = inputs.input_ids[0].argmax().item()
    return F.normalize(out[0, eos_idx].unsqueeze(0), dim=-1).squeeze()


def sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ── Experiment 1: Negation ────────────────────────────────────────────────────
# Finding: neither CLIP nor sentence transformers reliably understand negation.
# "not X" stays closest to X because the model learned image-text alignment —
# a photo described as "not happy" visually looks like any other emotional scene.

print("=" * 60)
print("  Exp 1 — Does 'not X' shift towards the antonym?")
print("=" * 60)

pairs = [
    ("not happy",     "happy",     "sad"),
    ("not hot",       "hot",       "cold"),
    ("not big",       "big",       "small"),
    ("not beautiful", "beautiful", "ugly"),
    ("not fast",      "fast",      "slow"),
]

for phrase, word, antonym in pairs:
    e_p, e_w, e_a = get_text_embedding(phrase), get_text_embedding(word), get_text_embedding(antonym)
    winner = "antonym ✓" if sim(e_p, e_a) > sim(e_p, e_w) else "original ✗"
    delta  = sim(e_p, e_a) - sim(e_p, e_w)
    print(f"\n  '{phrase}'")
    print(f"    sim('{word}'):    {sim(e_p, e_w):.4f}")
    print(f"    sim('{antonym}'): {sim(e_p, e_a):.4f}  → {winner}  Δ={delta:+.3f}")


# ── Experiment 2: Phrase modifiers ────────────────────────────────────────────
# Finding: modifiers can pull the embedding in the expected direction ("a small
# dog" → closer to "puppy"), but the base noun typically dominates. CLIP gives
# more weight to visually salient keywords than to surrounding context.

print("\n\n" + "=" * 60)
print("  Exp 2 — Do modifiers shift the embedding?")
print("=" * 60)

cases = [
    ("dog",   "a small dog",         "puppy"),
    ("dog",   "a dangerous dog",     "wolf"),
    ("fire",  "fire in a fireplace", "cozy"),
    ("fire",  "fire in a building",  "danger"),
    ("light", "dim light",           "dark"),
    ("light", "bright light",        "sun"),
]

for base, phrase, comparison in cases:
    e_base   = get_text_embedding(base)
    e_phrase = get_text_embedding(phrase)
    e_comp   = get_text_embedding(comparison)
    shifted = sim(e_phrase, e_comp) > sim(e_phrase, e_base)
    result  = "modifier won" if shifted else "base noun dominated"
    print(f"\n  '{phrase}'")
    print(f"    sim('{base}'):       {sim(e_phrase, e_base):.4f}")
    print(f"    sim('{comparison}'): {sim(e_phrase, e_comp):.4f}  → {result}")


# ── Experiment 3: 3D phrase map ───────────────────────────────────────────────
# Phrases with opposite emotional valence cluster on opposite sides.
# Neutral phrases land in a separate region, away from both groups.

print("\n\n" + "=" * 60)
print("  Exp 3 — 3D map: positive / negative / neutral phrases")
print("=" * 60)

items = {
    "positive": [
        "happy", "I feel wonderful", "pure happiness",
        "feeling loved", "joyful", "so grateful",
    ],
    "negative": [
        "sad", "I feel terrible", "deep sadness",
        "filled with rage", "scared", "broken hearted",
    ],
    "neutral": [
        "the weather is cloudy", "a table and two chairs",
        "the file was saved", "turn left at the corner",
    ],
}

all_texts  = [t for ts in items.values() for t in ts]
all_groups = [g for g, ts in items.items() for _ in ts]
colors_map = {"positive": "#e07b00", "negative": "#c0392b", "neutral": "#7f8c8d"}

print("  Generating embeddings...")
vecs = np.stack([get_text_embedding(t).detach().cpu().float().numpy() for t in all_texts])

pca = PCA(n_components=3)
coords = pca.fit_transform(vecs)
var = pca.explained_variance_ratio_
print(f"  PCA 3D: {sum(var)*100:.1f}% variance explained")

fig = go.Figure()
group_idxs = {}
for i, g in enumerate(all_groups):
    group_idxs.setdefault(g, []).append(i)

for group, idxs in group_idxs.items():
    c = coords[idxs]
    fig.add_trace(go.Scatter3d(
        x=c[:, 0], y=c[:, 1], z=c[:, 2],
        mode="markers+text",
        text=[all_texts[i] for i in idxs],
        textposition="top center",
        textfont=dict(size=8, color=colors_map[group]),
        marker=dict(size=7, color=colors_map[group], opacity=0.9,
                    line=dict(color="white", width=0.5)),
        name=group,
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))

fig.update_layout(
    title=dict(
        text=f"Phrase Embeddings — CLIP Text Encoder<br>"
             f"<sup>PCA 3D: {sum(var)*100:.1f}% variance | Drag to rotate</sup>",
        font=dict(size=14),
    ),
    scene=dict(
        xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
        yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
        zaxis_title=f"PC3 ({var[2]*100:.1f}%)",
        bgcolor="rgba(245,245,250,1)",
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    margin=dict(l=0, r=0, t=70, b=0),
    width=1000, height=700,
)
fig.write_html("output/03_phrase_map_3d.html")
fig.show()
print("  Saved to output/03_phrase_map_3d.html")
