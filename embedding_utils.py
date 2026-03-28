"""
Core library for CLIP token embedding exploration.

Loads the CLIP text encoder and exposes helper functions used by all
experiment scripts. Import from here rather than loading models repeatedly.

Public API:
  tokenizer, text_encoder, token_emb_layer, device
  ALL_EMBEDDINGS, ALL_EMBEDDINGS_NORM

  word_to_id(word)         → int
  get_embedding(word)      → Tensor [768]
  nearest_tokens(vec, ...)  → list[(word, sim)]

  exp_nearest(word)
  exp_opposite(word)
  exp_arithmetic(positive, negative)
  exp_visualize(word_groups)
  exp_density(words)
  exp_isolation(words)
  exp_density_map(word_groups)
  exp_region_map(seed_words)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel, logging

logging.set_verbosity_error()

# ─── Model loading ────────────────────────────────────────────────────────────

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
text_encoder.eval()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device: {device}")
text_encoder = text_encoder.to(device)

# Full token embedding table: shape [49408, 768]
# Row i is the raw vector for token ID i — no context, pure lookup.
token_emb_layer = text_encoder.text_model.embeddings.token_embedding
ALL_EMBEDDINGS = token_emb_layer.weight.data
ALL_EMBEDDINGS_NORM = torch.nn.functional.normalize(ALL_EMBEDDINGS, dim=-1)


# ─── Base helpers ─────────────────────────────────────────────────────────────

def word_to_id(word: str) -> int:
    """Return the token ID for a word. Warns if the word spans multiple tokens."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if len(ids) != 1:
        print(f"  ⚠  '{word}' tokenizes to {len(ids)} tokens: {ids} — using first only")
    return ids[0]


def get_embedding(word: str) -> torch.Tensor:
    """Raw token embedding vector [768] for a single word."""
    return token_emb_layer(torch.tensor(word_to_id(word), device=device))


def nearest_tokens(vec: torch.Tensor, n: int = 10, exclude: list = []) -> list:
    """
    Find the n tokens closest to `vec` by cosine similarity.
    Returns a list of (decoded_word, similarity) tuples.
    """
    vec_norm = torch.nn.functional.normalize(vec.unsqueeze(0), dim=-1)
    sims = (ALL_EMBEDDINGS_NORM.to(device) @ vec_norm.T).squeeze()

    for w in exclude:
        try:
            sims[word_to_id(w)] = -999
        except Exception:
            pass

    top_ids = sims.topk(n + 50).indices.tolist()
    results, seen = [], set()
    for tid in top_ids:
        if tokenizer.convert_ids_to_tokens(tid).startswith("<|"):
            continue
        word_decoded = tokenizer.decode([tid]).strip()
        if not word_decoded or word_decoded in seen:
            continue
        seen.add(word_decoded)
        results.append((word_decoded, round(sims[tid].item(), 4)))
        if len(results) >= n:
            break
    return results


# ─── Experiment functions ─────────────────────────────────────────────────────

def exp_nearest(word: str, n: int = 8):
    """Print the n tokens closest to `word` in the embedding space."""
    print(f"\n{'─'*52}")
    print(f"  Nearest tokens to '{word}'")
    print(f"{'─'*52}")
    for w, s in nearest_tokens(get_embedding(word), n=n, exclude=[word]):
        print(f"  {w:22s}  {s:.4f}  {'█' * int(s * 32)}")


def exp_opposite(word: str, n: int = 8):
    """
    Print nearest tokens to the negated vector (-1 * embedding).

    The geometric opposite is the diametrically opposite point on the unit
    sphere. This is NOT guaranteed to be the semantic antonym — the embedding
    space is asymmetric. In practice the opposite direction lands in a sparse
    region populated by rare or fragmented tokens.
    """
    print(f"\n{'─'*52}")
    print(f"  Geometric opposite of '{word}'  (-1 × embedding)")
    print(f"{'─'*52}")
    for w, s in nearest_tokens(-get_embedding(word), n=n, exclude=[word]):
        print(f"  {w:22s}  {s:.4f}  {'█' * int(s * 32)}")


def exp_arithmetic(positive: list, negative: list = [], n: int = 8):
    """
    Vector addition and subtraction over embeddings.

    If the space captured structured semantic relationships, linear operations
    should preserve them. Classic example: king - man + woman ≈ queen.
    CLIP embeddings are optimised for image-text alignment, not pure text
    semantics, so analogies work partially but not as strongly as Word2Vec.
    """
    expr = " + ".join(positive) + ((" - " + " - ".join(negative)) if negative else "")
    print(f"\n{'─'*52}")
    print(f"  Arithmetic:  {expr}")
    print(f"{'─'*52}")
    result = sum(get_embedding(w) for w in positive) - sum(get_embedding(w) for w in negative)
    for w, s in nearest_tokens(result, n=n, exclude=positive + negative):
        print(f"  {w:22s}  {s:.4f}  {'█' * int(s * 32)}")


def exp_visualize(word_groups: dict, title: str = "Token Embeddings — PCA 2D",
                  save_path: str = "embeddings_pca.png"):
    """
    Project word groups onto a 2D PCA plane and save as PNG.

    PCA finds the two directions of greatest variance in 768-dimensional space.
    Semantically related words should cluster together if the space is well-
    structured. The explained variance tells you how much information was lost
    in the compression.
    """
    from sklearn.decomposition import PCA

    all_words, all_vecs, groups = [], [], []
    for group_name, words in word_groups.items():
        for w in words:
            try:
                all_words.append(w)
                all_vecs.append(get_embedding(w).detach().cpu().float().numpy())
                groups.append(group_name)
            except Exception as e:
                print(f"  Skipping '{w}': {e}")

    coords = PCA(n_components=2).fit_transform(np.stack(all_vecs))
    pca = PCA(n_components=2).fit(np.stack(all_vecs))
    coords = pca.transform(np.stack(all_vecs))
    var = pca.explained_variance_ratio_.sum() * 100
    print(f"\n  PCA 2D: {var:.1f}% variance explained")

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = plt.cm.tab10.colors
    group_names = list(word_groups.keys())

    for word, (x, y), group in zip(all_words, coords, groups):
        color = colors[group_names.index(group) % 10]
        ax.scatter(x, y, color=color, s=90, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(word, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    for i, g in enumerate(group_names):
        ax.scatter([], [], color=colors[i % 10], label=g, s=80)

    ax.legend(loc="best", framealpha=0.85, fontsize=9)
    ax.set_title(f"{title}\n(variance explained: {var:.1f}%)", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.4)
    ax.axvline(0, color="gray", linewidth=0.4)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")


def exp_density(words: list, thresholds: list = [0.2, 0.3, 0.4, 0.5]):
    """
    For each word, count how many vocabulary tokens fall within each cosine
    similarity threshold.

    Words in dense regions have many close neighbours — typically common,
    concrete, or polysemous concepts. Isolated words have few neighbours —
    typically rare, technical, or highly specific terms.
    """
    print(f"\n{'─'*62}")
    print(f"  Local density  (# vocab tokens within similarity threshold)")
    print(f"  {'Word':20s}", end="")
    for t in thresholds:
        print(f"  sim>{t}", end="")
    print(f"\n{'─'*62}")

    all_embs_norm = ALL_EMBEDDINGS_NORM.to(device)
    for word in words:
        vec_norm = torch.nn.functional.normalize(get_embedding(word).unsqueeze(0), dim=-1)
        sims = (all_embs_norm @ vec_norm.T).squeeze()
        counts = [int((sims > t).sum().item()) - 1 for t in thresholds]
        print(f"  {word:20s}", end="")
        for c in counts:
            print(f"  {c:>6}", end="")
        print()


def exp_isolation(words: list, n_neighbors: int = 5):
    """
    For each word in the list, compute the average similarity to its
    n closest neighbours *within the list*.

    High score → word is central to the group.
    Low score  → word is a semantic outlier.
    """
    print(f"\n{'─'*52}")
    print(f"  Isolation within group (avg of {n_neighbors} nearest neighbours)")
    print(f"{'─'*52}")

    vecs = torch.cat([
        torch.nn.functional.normalize(get_embedding(w).unsqueeze(0), dim=-1)
        for w in words
    ])
    sim_matrix = (vecs @ vecs.T).cpu()

    scores = []
    for i, word in enumerate(words):
        row = sim_matrix[i].clone()
        row[i] = -1
        avg_sim = row.topk(min(n_neighbors, len(words) - 1)).values.mean().item()
        scores.append((word, avg_sim))
    scores.sort(key=lambda x: x[1], reverse=True)

    max_s, min_s = scores[0][1], scores[-1][1]
    for word, s in scores:
        bar_len = int(((s - min_s) / (max_s - min_s + 1e-9)) * 28)
        bar = "█" * bar_len + "░" * (28 - bar_len)
        label = " ← most central" if s == max_s else (" ← most isolated" if s == min_s else "")
        print(f"  {word:20s}  {s:.4f}  {bar}{label}")


def exp_density_map(word_groups: dict, vocab_sample: int = 3000,
                    save_path: str = "embeddings_density.png"):
    """
    PCA map where point SIZE encodes local density (number of vocab neighbours
    within cosine similarity > 0.25, estimated from a random vocab sample).

    Large points = dense region (common/central concept).
    Small points = isolated concept.
    """
    from sklearn.decomposition import PCA

    all_words, all_vecs, groups = [], [], []
    for group_name, words in word_groups.items():
        for w in words:
            try:
                all_words.append(w)
                all_vecs.append(get_embedding(w).detach().cpu().float().numpy())
                groups.append(group_name)
            except Exception:
                pass

    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.stack(all_vecs))
    var = pca.explained_variance_ratio_.sum() * 100

    print(f"\n  Estimating densities from {vocab_sample} vocab tokens...")
    sample_ids = torch.randperm(ALL_EMBEDDINGS.shape[0])[:vocab_sample]
    sample_embs = torch.nn.functional.normalize(ALL_EMBEDDINGS[sample_ids].to(device), dim=-1)

    densities = []
    for w in all_words:
        vec_norm = torch.nn.functional.normalize(get_embedding(w).unsqueeze(0), dim=-1)
        sims = (sample_embs @ vec_norm.T).squeeze()
        densities.append(int((sims > 0.25).sum().item()))

    d_arr = np.array(densities, dtype=float)
    d_norm = (d_arr - d_arr.min()) / (d_arr.max() - d_arr.min() + 1e-9)
    sizes = 40 + d_norm * 360

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.tab10.colors
    group_names = list(word_groups.keys())

    for word, (x, y), group, size, density in zip(all_words, coords, groups, sizes, densities):
        color = colors[group_names.index(group) % 10]
        ax.scatter(x, y, color=color, s=size, zorder=3, edgecolors="white", linewidths=0.6, alpha=0.85)
        ax.annotate(f"{word}\n({density})", (x, y), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5)

    for i, g in enumerate(group_names):
        ax.scatter([], [], color=colors[i % 10], label=g, s=80)
    for d_val, label in [(d_arr.min(), "isolated"), (d_arr.mean(), "average"), (d_arr.max(), "dense")]:
        s = 40 + ((d_val - d_arr.min()) / (d_arr.max() - d_arr.min() + 1e-9)) * 360
        ax.scatter([], [], color="gray", s=s, label=f"density: {label}", alpha=0.6)

    ax.legend(loc="best", framealpha=0.85, fontsize=8)
    ax.set_title(
        f"Embedding Space Density Map\n"
        f"Point size = # vocab neighbours | PCA {var:.1f}%", fontsize=11
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved to {save_path}")


def exp_region_map(seed_words: list, top_n: int = 60, threshold: float = 0.18,
                   title: str = "Semantic Region", save_path: str = "region_map.png"):
    """
    Define a semantic region from anchor words, discover vocabulary tokens that
    fall within it, and visualise everything on a 2D PCA map.

    Steps:
      1. Compute the centroid (mean) of seed word vectors.
      2. Scan all 49,408 vocab tokens and rank by similarity to the centroid.
      3. Collect the top_n tokens above the threshold.
      4. Project seeds + discovered tokens via PCA 2D.
      5. Draw a convex hull around the seed cluster.
    """
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull

    print(f"\n  Computing centroid for: {seed_words}")
    seed_vecs = torch.stack([get_embedding(w) for w in seed_words])
    centroid = seed_vecs.mean(dim=0)
    centroid_norm = torch.nn.functional.normalize(centroid.unsqueeze(0), dim=-1)

    print(f"  Scanning {ALL_EMBEDDINGS.shape[0]:,} vocab tokens...")
    sims = (ALL_EMBEDDINGS_NORM.to(device) @ centroid_norm.T).squeeze()

    seed_ids = {word_to_id(w) for w in seed_words}
    disc_words, disc_sims, seen = [], [], set(seed_words)
    for tid in sims.topk(top_n + len(seed_ids) + 50).indices.tolist():
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

    print(f"  Found {len(disc_words)} tokens in the region (sim ≥ {threshold})")
    print(f"\n  Top 20 discovered:")
    for w, s in zip(disc_words[:20], disc_sims[:20]):
        print(f"    {w:22s}  {s:.4f}  {'█' * int(s * 40)}")

    all_vecs_np = [get_embedding(w).detach().cpu().float().numpy() for w in seed_words]
    for w in disc_words:
        tid = tokenizer.encode(w, add_special_tokens=False)[0]
        all_vecs_np.append(ALL_EMBEDDINGS[tid].float().cpu().numpy())

    pca = PCA(n_components=2)
    coords = pca.fit_transform(np.stack(all_vecs_np))
    var = pca.explained_variance_ratio_.sum() * 100

    seed_coords = coords[:len(seed_words)]
    disc_coords = coords[len(seed_words):]
    disc_sims_arr = np.array(disc_sims)

    fig, ax = plt.subplots(figsize=(13, 10))
    sc = ax.scatter(disc_coords[:, 0], disc_coords[:, 1], c=disc_sims_arr, cmap="YlOrRd",
                    s=55, alpha=0.75, zorder=2, edgecolors="white", linewidths=0.4,
                    vmin=threshold, vmax=disc_sims_arr.max())
    plt.colorbar(sc, ax=ax, label="Similarity to centroid", pad=0.01)
    for word, (x, y) in zip(disc_words, disc_coords):
        ax.annotate(word, (x, y), textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color="#444444")

    ax.scatter(seed_coords[:, 0], seed_coords[:, 1], color="#1a3a6b", s=180,
               zorder=5, edgecolors="white", linewidths=1.2, label="Anchor seeds")
    for word, (x, y) in zip(seed_words, seed_coords):
        ax.annotate(word, (x, y), textcoords="offset points", xytext=(6, 5),
                    fontsize=10, fontweight="bold", color="#1a3a6b")

    centroid_2d = pca.transform(centroid.detach().cpu().float().numpy().reshape(1, -1))
    ax.scatter(centroid_2d[0, 0], centroid_2d[0, 1], marker="X", color="black",
               s=200, zorder=6, label="Centroid")

    if len(seed_coords) >= 3:
        try:
            hull = ConvexHull(seed_coords)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.plot(seed_coords[hull_pts, 0], seed_coords[hull_pts, 1],
                    "b--", linewidth=1.5, alpha=0.5, label="Anchor region (convex hull)")
            ax.fill(seed_coords[hull.vertices, 0], seed_coords[hull.vertices, 1],
                    alpha=0.07, color="blue")
        except Exception:
            pass

    ax.legend(loc="lower right", framealpha=0.85, fontsize=9)
    ax.set_title(
        f"{title}\nSeeds: {seed_words}\n"
        f"{len(disc_words)} discovered tokens | PCA {var:.1f}% variance", fontsize=10
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Saved to {save_path}")
