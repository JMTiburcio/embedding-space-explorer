"""
Experiment 04 — CLIP vs ModernBERT Embedding Comparison

Compares two embedding models on the same tasks:
  - CLIP (openai/clip-vit-large-patch14): trained for image-text alignment.
  - ModernBERT (nomic-ai/modernbert-embed-base): trained for text-text
    semantic similarity with a contrastive objective.

Experiments:
  1. Negation — which model better understands "not X" → antonym?
  2. Analogies — king - man + woman = queen?
  3. 3D map — same phrases visualised in both embedding spaces side by side.

Key findings:
  - Both models handle analogies reasonably well.
  - Neither model reliably handles logical negation.
  - CLIP slightly outperforms ModernBERT on negation for adjectives ("not big",
    "not beautiful"), but fails on verbs/states ("not happy", "not fast").
  - ModernBERT clusters semantically related phrases more tightly (higher
    intra-group similarity), which *hurts* negation: "not happy" is about
    happiness, so it lands close to "happy".

Run:
  python 04_model_comparison.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ── Load models ───────────────────────────────────────────────────────────────

print("Loading CLIP...")
clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)

print("Loading ModernBERT...")
mb_tok = AutoTokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
mb_model = AutoModel.from_pretrained("nomic-ai/modernbert-embed-base").eval().to(device)


def clip_embed(text: str) -> torch.Tensor:
    """CLIP: EOS token vector, L2-normalised. [768]"""
    inputs = clip_tok(text, padding="max_length", max_length=clip_tok.model_max_length,
                      truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = clip_model(inputs.input_ids.to(device))[0]
    eos_idx = inputs.input_ids[0].argmax().item()
    return F.normalize(out[0, eos_idx].unsqueeze(0), dim=-1).squeeze()


def modernbert_embed(text: str) -> torch.Tensor:
    """
    ModernBERT: mean-pooled last hidden state, L2-normalised. [768]
    The "clustering: " prefix signals the model to optimise for
    semantic similarity rather than retrieval asymmetry.
    """
    inputs = mb_tok(f"clustering: {text}", return_tensors="pt",
                    truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = mb_model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    return F.normalize(pooled.squeeze(0), dim=-1)


def sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ── Experiment 1: Negation ────────────────────────────────────────────────────

negation_pairs = [
    ("not happy",     "happy",     "sad"),
    ("not hot",       "hot",       "cold"),
    ("not big",       "big",       "small"),
    ("not beautiful", "beautiful", "ugly"),
    ("not fast",      "fast",      "slow"),
    ("not safe",      "safe",      "dangerous"),
]

print("=" * 70)
print("  Exp 1 — Negation  (✓ = antonym closer, ✗ = original closer)")
print("=" * 70)
print(f"\n  {'Phrase':20s}  {'→':8s}  {'CLIP':>24s}  {'ModernBERT':>24s}")
print(f"  {'─'*20}  {'─'*8}  {'─'*24}  {'─'*24}")

for phrase, word, antonym in negation_pairs:
    c_p, c_w, c_a = clip_embed(phrase), clip_embed(word), clip_embed(antonym)
    clip_result = "✓ antonym" if sim(c_p, c_a) > sim(c_p, c_w) else "✗ original"
    clip_delta  = sim(c_p, c_a) - sim(c_p, c_w)

    m_p, m_w, m_a = modernbert_embed(phrase), modernbert_embed(word), modernbert_embed(antonym)
    mb_result = "✓ antonym" if sim(m_p, m_a) > sim(m_p, m_w) else "✗ original"
    mb_delta  = sim(m_p, m_a) - sim(m_p, m_w)

    print(f"  {phrase:20s}  {'→'+antonym:8s}  "
          f"{clip_result:10s} Δ={clip_delta:+.3f}  "
          f"{mb_result:10s} Δ={mb_delta:+.3f}")


# ── Experiment 2: Analogies ───────────────────────────────────────────────────

CANDIDATES = [
    "queen", "king", "princess", "prince", "woman", "man",
    "berlin", "paris", "london", "madrid", "rome", "vienna",
    "warm", "hot", "cold", "cool", "mild", "tepid",
    "dusk", "dawn", "day", "night", "evening", "morning",
    "kitten", "puppy", "cat", "dog", "cub", "foal",
]

def nearest_candidate(vec, used, embed_fn):
    best_label, best_sim = None, -999
    for label in CANDIDATES:
        if label in used:
            continue
        s = sim(vec, embed_fn(label))
        if s > best_sim:
            best_sim, best_label = s, label
    return best_label, best_sim

analogy_tests = [
    (["king", "woman"],    ["man"],    "queen"),
    (["paris", "germany"], ["france"], "berlin"),
    (["hot", "cold"],      [],        "warm"),
    (["day", "night"],     [],        "dusk"),
    (["puppy", "cat"],     ["dog"],   "kitten"),
]

print("\n\n" + "=" * 70)
print("  Exp 2 — Analogies  (✓ = expected answer, ✗ = different answer)")
print("=" * 70)
print(f"\n  {'Expression':32s}  {'Expected':8s}  {'CLIP':>12s}  {'ModernBERT':>12s}")
print(f"  {'─'*32}  {'─'*8}  {'─'*12}  {'─'*12}")

for positive, negative, expected in analogy_tests:
    expr = " + ".join(positive) + ((" - " + " - ".join(negative)) if negative else "")
    used = set(positive + negative)

    c_vec = sum(clip_embed(w) for w in positive) - sum(clip_embed(w) for w in negative)
    c_vec = F.normalize(c_vec.unsqueeze(0), dim=-1).squeeze()
    c_near, _ = nearest_candidate(c_vec, used, clip_embed)

    m_vec = sum(modernbert_embed(w) for w in positive) - sum(modernbert_embed(w) for w in negative)
    m_vec = F.normalize(m_vec.unsqueeze(0), dim=-1).squeeze()
    m_near, _ = nearest_candidate(m_vec, used, modernbert_embed)

    print(f"  {expr:32s}  {expected:8s}  "
          f"{'✓' if c_near == expected else '✗'} {c_near:10s}  "
          f"{'✓' if m_near == expected else '✗'} {m_near:10s}")


# ── Experiment 3: 3D side-by-side map ────────────────────────────────────────

print("\n\n" + "=" * 70)
print("  Exp 3 — 3D side-by-side: same phrases in both spaces")
print("=" * 70)

phrase_groups = {
    "positive": ["happy", "I feel wonderful", "pure happiness", "feeling loved", "joyful"],
    "negative": ["sad", "I feel terrible", "deep sadness", "filled with rage", "scared"],
    "neutral":  ["the weather is cloudy", "a table and two chairs",
                 "the file was saved", "turn left at the corner"],
}

all_texts  = [t for ts in phrase_groups.values() for t in ts]
all_groups = [g for g, ts in phrase_groups.items() for _ in ts]
colors_map = {"positive": "#e07b00", "negative": "#c0392b", "neutral": "#7f8c8d"}

print("  Generating CLIP embeddings...")
clip_vecs = np.stack([clip_embed(t).detach().cpu().float().numpy() for t in all_texts])
print("  Generating ModernBERT embeddings...")
mb_vecs = np.stack([modernbert_embed(t).detach().cpu().float().numpy() for t in all_texts])

def to_3d(vecs):
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vecs)
    return coords, pca.explained_variance_ratio_

clip_coords, clip_var = to_3d(clip_vecs)
mb_coords,   mb_var   = to_3d(mb_vecs)
print(f"  CLIP PCA 3D:       {sum(clip_var)*100:.1f}% variance")
print(f"  ModernBERT PCA 3D: {sum(mb_var)*100:.1f}% variance")

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    subplot_titles=[
        f"CLIP  ({sum(clip_var)*100:.1f}% variance)",
        f"ModernBERT  ({sum(mb_var)*100:.1f}% variance)",
    ],
)

group_idxs = {}
for i, g in enumerate(all_groups):
    group_idxs.setdefault(g, []).append(i)

for col, coords in enumerate([clip_coords, mb_coords], start=1):
    for group, idxs in group_idxs.items():
        c = coords[idxs]
        fig.add_trace(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode="markers+text",
            text=[all_texts[i] for i in idxs],
            textposition="top center",
            textfont=dict(size=8, color=colors_map[group]),
            marker=dict(size=6, color=colors_map[group], opacity=0.9,
                        line=dict(color="white", width=0.5)),
            name=group,
            showlegend=(col == 1),
            legendgroup=group,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ), row=1, col=col)

fig.update_layout(
    title=dict(
        text="CLIP vs ModernBERT — same phrases, different spaces<br>"
             "<sup>Drag each subplot to rotate independently</sup>",
        font=dict(size=14),
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
    margin=dict(l=0, r=0, t=80, b=0),
    width=1300, height=650,
)
fig.write_html("output/04_model_comparison_3d.html")
fig.show()
print("  Saved to output/04_model_comparison_3d.html")
