"""
Experiment 02 — Semantic Region Mapping

Given a set of anchor (seed) words, this experiment:
  1. Computes the centroid of their embeddings — the semantic "centre of gravity".
  2. Scans the full 49,408-token CLIP vocabulary for tokens near that centroid.
  3. Visualises the discovered region in both 2D (static PNG) and 3D (interactive HTML).

Key insight: the centroid cancels out what is specific to each seed word and
retains what they all share. Tokens discovered near the centroid occupy the
same semantic neighbourhood without necessarily being similar to any single seed.

Run:
  python 02_semantic_region.py
"""

import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

from embedding_utils import (
    tokenizer, token_emb_layer, ALL_EMBEDDINGS, ALL_EMBEDDINGS_NORM,
    device, word_to_id, get_embedding, exp_region_map,
)


def region_map_3d(seed_words: list, top_n: int = 80, threshold: float = 0.18,
                  title: str = "Semantic Region 3D", save_path: str = "region_3d.html"):
    """
    Interactive 3D version of exp_region_map.

    PCA with 3 components captures more variance than 2D, giving a more
    faithful representation of the 768-dimensional space.

    The HTML output opens in the browser and supports rotation, zoom, and hover.
    """
    print(f"\n  Computing centroid for: {seed_words}")
    seed_vecs = torch.stack([get_embedding(w) for w in seed_words])
    centroid = seed_vecs.mean(dim=0)
    centroid_norm = F.normalize(centroid.unsqueeze(0), dim=-1)

    print(f"  Scanning {ALL_EMBEDDINGS.shape[0]:,} vocab tokens...")
    sims = (ALL_EMBEDDINGS_NORM.to(device) @ centroid_norm.T).squeeze()

    seed_ids = {word_to_id(w) for w in seed_words}
    disc_words, disc_sims, seen = [], [], set(seed_words)
    for tid in sims.topk(top_n + len(seed_ids) + 100).indices.tolist():
        if tid in seed_ids:
            continue
        s = sims[tid].item()
        if s < threshold:
            break
        word_decoded = tokenizer.decode([tid]).strip()
        raw = tokenizer.convert_ids_to_tokens(tid)
        if not word_decoded or word_decoded in seen or raw.startswith("<|"):
            continue
        seen.add(word_decoded)
        disc_words.append(word_decoded)
        disc_sims.append(round(s, 4))
        if len(disc_words) >= top_n:
            break

    print(f"  Found {len(disc_words)} tokens (sim ≥ {threshold})")

    all_vecs_np = [get_embedding(w).detach().cpu().float().numpy() for w in seed_words]
    for w in disc_words:
        tid = tokenizer.encode(w, add_special_tokens=False)[0]
        all_vecs_np.append(ALL_EMBEDDINGS[tid].float().cpu().numpy())
    all_vecs_np.append(centroid.detach().cpu().float().numpy())

    pca = PCA(n_components=3)
    coords = pca.fit_transform(np.stack(all_vecs_np))
    var = pca.explained_variance_ratio_
    print(f"  PCA 3D: {sum(var)*100:.1f}% variance explained")

    seed_c = coords[:len(seed_words)]
    disc_c = coords[len(seed_words):-1]
    cent_c = coords[-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=disc_c[:, 0], y=disc_c[:, 1], z=disc_c[:, 2],
        mode="markers+text",
        text=disc_words,
        textposition="top center",
        textfont=dict(size=8, color="#555555"),
        marker=dict(size=5, color=disc_sims, colorscale="YlOrRd",
                    cmin=threshold, cmax=max(disc_sims),
                    colorbar=dict(title="Similarity<br>to centroid", thickness=14),
                    opacity=0.8),
        name="Discovered tokens",
        hovertemplate="<b>%{text}</b><br>sim: %{marker.color:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=seed_c[:, 0], y=seed_c[:, 1], z=seed_c[:, 2],
        mode="markers+text",
        text=seed_words,
        textposition="top center",
        textfont=dict(size=11, color="#1a3a6b", family="Arial Black"),
        marker=dict(size=10, color="#1a3a6b", line=dict(color="white", width=1.5)),
        name="Anchor seeds",
        hovertemplate="<b>%{text}</b><br>seed<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[cent_c[0]], y=[cent_c[1]], z=[cent_c[2]],
        mode="markers+text",
        text=["centroid"],
        textposition="top center",
        textfont=dict(size=10, color="black"),
        marker=dict(size=12, symbol="cross", color="black"),
        name="Centroid",
    ))
    for i, w in enumerate(seed_words):
        fig.add_trace(go.Scatter3d(
            x=[seed_c[i, 0], cent_c[0]], y=[seed_c[i, 1], cent_c[1]],
            z=[seed_c[i, 2], cent_c[2]],
            mode="lines",
            line=dict(color="#1a3a6b", width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{len(disc_words)} tokens | "
                 f"PCA 3D: {sum(var)*100:.1f}% variance | Drag to rotate</sup>",
            font=dict(size=14),
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({var[1]*100:.1f}%)",
            zaxis_title=f"PC3 ({var[2]*100:.1f}%)",
            bgcolor="rgba(245,245,250,1)",
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=60, b=0),
        width=1100, height=750,
    )
    fig.write_html(save_path)
    fig.show()
    print(f"  Saved to {save_path}")


# ── Run ───────────────────────────────────────────────────────────────────────

SEEDS = ["happy", "sad", "angry", "fear", "love", "joy"]

# 2D static map
exp_region_map(
    seed_words=SEEDS,
    top_n=60,
    threshold=0.18,
    title="Semantic Region: emotions",
    save_path="output/02_region_emotions_2d.png",
)

# 3D interactive map
region_map_3d(
    seed_words=SEEDS,
    top_n=80,
    threshold=0.18,
    title="Semantic Region 3D: emotions",
    save_path="output/02_region_emotions_3d.html",
)

# ── Try other regions (uncomment to explore) ──────────────────────────────────

# region_map_3d(
#     seed_words=["king", "queen", "prince", "princess", "throne"],
#     title="Semantic Region 3D: royalty",
#     save_path="output/02_region_royalty_3d.html",
# )

# region_map_3d(
#     seed_words=["red", "blue", "green", "yellow", "purple", "orange"],
#     title="Semantic Region 3D: colors",
#     save_path="output/02_region_colors_3d.html",
# )
