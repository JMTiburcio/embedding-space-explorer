# Embedding Space Explorer

A hands-on exploration of token and text embeddings using CLIP and ModernBERT. Each experiment peels back a layer of how language models represent meaning as vectors, starting from individual tokens and building up to full phrases and model comparisons.

The original motivation came from a [Stable Diffusion deep dive notebook](stable_diffusion_deep_dive.py) that shows how text prompts travel through a tokenizer → text encoder → diffusion loop to generate images.

---

## Setup

```bash
conda env create -f environment.yml
conda activate embedding-space-explorer
```

Models are downloaded automatically from HuggingFace on first run (~1 GB total).

---

## Experiments

### [01 — Token Exploration](01_token_exploration.py)

Explores the raw token embedding table of CLIP (49,408 tokens × 768 dimensions).

```bash
python 01_token_exploration.py
```

**What it does:**
- **Nearest neighbours**: finds the closest tokens to any word by cosine similarity.
- **Geometric opposite**: computes −1 × embedding and finds what lives there.
- **Vector arithmetic**: `king − man + woman`, `hot + cold`, geographic analogies.
- **Local density**: counts how many vocab tokens fall within similarity thresholds.
- **Intra-group isolation**: ranks which word is the semantic outlier in a list.
- **PCA 2D visualisation**: projects word groups onto a 2D plane.
- **Density map**: PCA where point size encodes local density.

**Key findings:**

| Finding | Detail |
|---|---|
| BPE duplicates | `"king"` maps to two different tokens: one as a word fragment (used in "kingdom") and one as a complete word. They have ~0.67 cosine similarity between them — this is why `"king"` appears in its own nearest-neighbours list with less than 1.0 similarity. |
| Geometric opposite ≠ antonym | `−hot` does not land near `"cold"`. The opposite direction of common words falls into a sparse region of rare/fragmented tokens. The space is asymmetric. |
| Arithmetic works partially | `king − man + woman` → **queen** ✓. `hot + cold` → **warm** ✓ (the midpoint of opposites lands at "lukewarm" temperature semantics). `paris − france + germany` → **berlin** ✓. |
| Density reflects frequency | `"the"` has 166 neighbours at sim > 0.2. `"serendipity"` has 0 neighbours at sim > 0.3. Common words occupy dense, crowded regions; rare words are isolated. |
| Outlier detection works | Adding `"airplane"` to a list of animals makes it the clear outlier. `"joy"` is surprisingly isolated among emotion words — likely because "Joy" also appears as a proper noun in training data. |

---

### [02 — Semantic Region Mapping](02_semantic_region.py)

Defines a semantic region from anchor words and discovers which vocabulary tokens fall inside it.

```bash
python 02_semantic_region.py
```

**How it works:**
1. Computes the **centroid** (vector mean) of the anchor words — the semantic centre of gravity.
2. Scans all 49,408 vocabulary tokens and ranks them by similarity to the centroid.
3. Visualises the discovered tokens in 2D (PNG) and 3D interactive HTML.

**Key findings:**

- The centroid cancels out what is specific to each seed and retains what they share. Seeds `["happy", "sad", "angry", "fear", "love", "joy"]` produce a centroid representing *emotional intensity* — not positive or negative affect specifically.
- The 60 discovered tokens include: `unhappy`, `luv`, `hate`, `pleased`, `hates`, `scared`, `crying`, `feeling`, `glad`, `proud`, `excited`, `shame`. This confirms the model clusters emotional vocabulary together regardless of valence.
- `"hate"` appears near `"love"` because both convey strong emotion — the space encodes *intensity*, not *sign*.
- The 3D interactive map (HTML) allows rotating the embedding cloud to find angular separations invisible in 2D.

---

### [03 — Phrase Embeddings](03_phrase_embeddings.py)

Shows that embeddings can be computed for any string (not just single words) and explores how context changes a vector.

```bash
python 03_phrase_embeddings.py
```

**The two levels of embedding:**

| Level | API | Context-aware? |
|---|---|---|
| Token embedding | `token_emb_layer(token_id)` | No — fixed lookup table |
| Text embedding | `text_encoder(tokenized_string)` | Yes — transformer attention over the full sequence |

**Key findings:**

| Experiment | Finding |
|---|---|
| Negation | `"not happy"` stays closest to `"happy"`, not `"sad"`. CLIP was trained for image-text alignment — a photo described as "not happy" looks like any emotional scene. Logical negation was never an objective. |
| Phrase modifiers | `"a small dog"` → closer to `"puppy"` than to `"dog"` ✓. The adjective pulled the vector in the right direction. However, `"fire in a building"` did not move towards `"danger"` — the base noun `"fire"` dominated. CLIP gives more weight to visually salient keywords than to narrative context. |
| 3D phrase map | Positive and negative emotional phrases cluster on opposite sides of the space. Neutral phrases form a clearly separate group. With full-sentence embeddings, PCA 3D explains ~40% of variance (vs ~10% for isolated tokens from a single semantic region). |

---

### [04 — Model Comparison: CLIP vs ModernBERT](04_model_comparison.py)

Runs the same experiments on two architecturally different models.

```bash
python 04_model_comparison.py
```

| Model | Training objective | Embedding method |
|---|---|---|
| `openai/clip-vit-large-patch14` | Align text ↔ image | [EOS] token vector |
| `nomic-ai/modernbert-embed-base` | Text-text semantic similarity | Mean pooling over all tokens |

**Results:**

**Negation:**

| Phrase | CLIP | ModernBERT |
|---|---|---|
| not happy | ✗ original | ✗ original |
| not hot | ✗ original | ✗ original |
| not big | ✓ antonym | ✗ original |
| not beautiful | ✓ antonym | ✗ original |
| not fast | ✗ original | ✗ original |
| not safe | ✗ original | ✗ original |

Neither model understands negation reliably. Surprisingly, CLIP performed better on adjectives. The reason ModernBERT fails *more* is that it was trained to bring semantically related content closer together — `"not happy"` and `"happy"` are topically related, so ModernBERT pulls them together even more strongly.

**Analogies:**

| Expression | Expected | CLIP | ModernBERT |
|---|---|---|---|
| king + woman − man | queen | ✓ | ✓ |
| paris + germany − france | berlin | ✓ | ✓ |
| hot + cold | warm | ✓ | ✓ |
| day + night | dusk | ✗ evening | ✗ morning |
| puppy + cat − dog | kitten | ✓ | ✓ |

Both models handle most analogies correctly. `day + night` is genuinely ambiguous — `evening` and `morning` are equally valid midpoints.

**3D comparison map:** The side-by-side HTML shows how each model organises the same set of phrases spatially. ModernBERT tends to produce tighter within-group clusters; CLIP spreads groups further apart.

---

## Why CLIP is not ideal for text-only experiments

CLIP's embedding space is optimised for a single task: given a text description, find the matching image (and vice versa). This means:

- Semantic similarity between texts is a *side effect*, not a training objective.
- Negation is invisible — "a dog" and "not a dog" both look like images of dogs.
- The embedding space is shaped by what is *visually distinguishable*, not what is *semantically distinct* in text.

For text-only work, models trained with text-text contrastive objectives (sentence-transformers, ModernBERT, E5, BGE) will produce more faithful semantic geometry.

---

## Project structure

```
embedding-space-explorer/
├── embedding_utils.py            # Core library: model loading + all exp_* functions
├── stable_diffusion_deep_dive.py # Reference notebook: SD pipeline from scratch
├── 01_token_exploration.py       # Token space: nearest, opposite, arithmetic, density
├── 02_semantic_region.py         # Centroid-based region discovery (2D + 3D)
├── 03_phrase_embeddings.py       # Full-sequence embeddings and context effects
├── 04_model_comparison.py        # CLIP vs ModernBERT side-by-side
├── environment.yml               # Conda environment
└── outputs/                      # Generated PNGs and HTMLs (git-ignored)
```

## Further reading

- [CLIP paper](https://arxiv.org/abs/2103.00020) — Learning Transferable Visual Models From Natural Language Supervision
- [ModernBERT paper](https://arxiv.org/abs/2412.13663) — Smarter, Better, Faster, Longer
- [Textual Inversion](https://textual-inversion.github.io/) — Learning to personalise text-to-image models
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Benchmark for text embedding models
- [Word2Vec paper](https://arxiv.org/abs/1301.3781) — Original vector arithmetic demonstrations
