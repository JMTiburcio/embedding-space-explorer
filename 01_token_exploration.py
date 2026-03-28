"""
Experiment 01 — Token Embedding Exploration

Explores the structure of CLIP's token embedding space using five lenses:
  1. Nearest neighbours   — what lives close to a word?
  2. Geometric opposite   — what is at -1 × embedding?
  3. Vector arithmetic    — does king - man + woman = queen?
  4. Local density        — which words are in dense vs sparse regions?
  5. Intra-group isolation — which word is the odd one out in a list?
  6. PCA visualisation    — do semantic groups cluster together?
  7. Density map          — PCA with point size encoding local density

Run:
  python 01_token_exploration.py
"""

from embedding_utils import (
    exp_nearest, exp_opposite, exp_arithmetic,
    exp_density, exp_isolation, exp_visualize, exp_density_map,
)

# ── 1. Nearest neighbours ─────────────────────────────────────────────────────
# NOTE: "king" will appear in the results with ~0.67 similarity instead of 1.0.
# This is because BPE creates two tokens for "king": one as a word fragment
# (used in "kingdom") and one as a complete word (with end-of-word marker).
# The exclude list removes only one; the other still shows up in results.
exp_nearest("king")
exp_nearest("happy")
exp_nearest("fire")

# ── 2. Geometric opposite ─────────────────────────────────────────────────────
# The region opposite to a common word is populated by rare, fragmented tokens.
# The space is asymmetric — common words cluster together, leaving the
# "negative" side of the sphere sparsely occupied.
exp_opposite("hot")
exp_opposite("king")
exp_opposite("beautiful")

# ── 3. Vector arithmetic ──────────────────────────────────────────────────────
exp_arithmetic(["king", "woman"], ["man"])        # classic Word2Vec test → queen
exp_arithmetic(["hot", "cold"])                   # midpoint of opposites → warm
exp_arithmetic(["paris", "germany"], ["france"])  # geographic analogy → berlin
exp_arithmetic(["day"], ["light"])                # concept without attribute

# ── 4. Local density ──────────────────────────────────────────────────────────
# "the" lives in a very dense region (166 neighbours at sim > 0.2).
# "serendipity" is almost completely isolated (0 neighbours at sim > 0.3).
exp_density([
    "the", "cat", "love", "algorithm",
    "photosynthesis", "red", "serendipity", "transformer",
])

# ── 5. Intra-group isolation ──────────────────────────────────────────────────
# "airplane" is the clear outlier in a list of animals.
# "joy" is surprisingly more isolated than "sad" among emotions —
# likely because "Joy" also appears as a proper name in the training data.
exp_isolation(["cat", "dog", "fish", "bird", "horse", "airplane"])
exp_isolation(["happy", "joy", "excited", "sad", "merry", "pleased"])

# ── 6. PCA visualisation ──────────────────────────────────────────────────────
exp_visualize(
    {
        "royalty":  ["king", "queen", "prince", "princess", "crown", "throne"],
        "animals":  ["cat", "dog", "horse", "fish", "bird", "wolf"],
        "emotions": ["happy", "sad", "angry", "fear", "love", "joy"],
        "weather":  ["hot", "cold", "rain", "snow", "wind", "storm"],
        "colors":   ["red", "blue", "green", "yellow", "purple", "orange"],
    },
    save_path="output/01_pca.png",
)

# ── 7. Density map ────────────────────────────────────────────────────────────
exp_density_map(
    {
        "royalty":  ["king", "queen", "prince", "princess", "crown", "throne"],
        "animals":  ["cat", "dog", "horse", "fish", "bird", "wolf"],
        "emotions": ["happy", "sad", "angry", "fear", "love", "joy"],
        "weather":  ["hot", "cold", "rain", "snow", "wind", "storm"],
        "colors":   ["red", "blue", "green", "yellow", "purple", "orange"],
    },
    save_path="output/01_density_map.png",
)
