# PRISM: a deferral monitor for personalized safety

PRISM is a lightweight input monitor that predicts, before a VLM answers, whether an
(image, query) pair needs the user's personal background to be answered safely. If it
does, the query should be deferred (ask for, or retrieve, the user's context) instead of
being answered generically.

PRISM reads frozen embeddings of the image and the query from an off-the-shelf
vision-language encoder and fuses them with bidirectional cross-modal modulation (Bi-FiLM):
the text embedding modulates the image embedding and vice versa. A sigmoid-gated, layer-normed
fused representation and a text residual feed the deferral head. Auxiliary heads for
image tone, text tone and image–text tone conflict are trained jointly.

All code in this directory is run from `prism_monitor/` (modules are imported as `model`
and `config`).

## Released checkpoints

Three PRISM models are released, one per frozen encoder. The table gives their results on the
paper's PRISM test split (1,080 queries). The PRISM training data (images and generated question
pairs), the train/val/test splits and the cached embeddings are not released;
[Training](#training) describes how to regenerate them.

| Config | Model | Encoder | Test AUROC | Test F1 |
|---|---|---|---|---|
| `default` | PRISM (SigLIP2) | `google/siglip2-large-patch16-512` | 0.983 | 0.932 |
| `clip_default` | PRISM (CLIP) | `openai/clip-vit-large-patch14` | 0.955 | 0.871 |
| `evaclip_default` | PRISM (EVA-CLIP) | `BAAI/EVA-CLIP-8B` | 0.978 | 0.925 |

Each checkpoint directory contains `model.safetensors` (the trainable head only; the
encoder is downloaded from the Hugging Face Hub when needed) and `config.json` (the
architecture and the encoder needed to rebuild the model). The encoder key `siglip_so400m`
in `config.json` and `ENCODER_REGISTRY` refers to SigLIP2-large
(`google/siglip2-large-patch16-512`).

**Checkpoint license.** The released weights are distributed under the MIT License.

**Pinned encoder revisions.** Every encoder download (`SiglipModel`, `CLIPModel`, the
Qwen3-VL model, their processors and tokenizers, and EVA-CLIP) passes `revision=` with the
Hub commit current when the paper's embeddings were computed (`revision` /
`processor_revision` in `ENCODER_REGISTRY`):

| Repository | Commit |
|---|---|
| `google/siglip2-large-patch16-512` | `49488218e80259885f3be61d7a9455faf833b7a8` |
| `openai/clip-vit-large-patch14` (also the EVA-CLIP processor) | `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| `BAAI/EVA-CLIP-8B` | `0e4dca944e8ece27eb9dfe4a488c0ed0c4644fc9` |
| `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` |

**Remote code.** `model/encoders.py` loads `BAAI/EVA-CLIP-8B` with `trust_remote_code=True` at the
pinned commit above.

**Qwen3-VL.** The code also defines `qwen_default`, PRISM on `Qwen/Qwen3-VL-2B-Instruct`, which
reached test AUROC 0.989 on the paper's test split. Its checkpoint is not released; train it with
the pipeline below. The Qwen3-VL encoder mean-pools the ViT and LM hidden states (2048-d) and maps
them to 1152-d with a frozen, randomly initialised linear projection. `cache_embeddings.py` saves
the projection and `train_models.py` copies it into the checkpoint, so the result runs end to end
from images.

## Download checkpoints

The checkpoints are on the Hugging Face Hub at
[`EdwardoSunny/multimodal-personalized-safety-prism`](https://huggingface.co/EdwardoSunny/multimodal-personalized-safety-prism).
From the repository root:

```bash
uv run python scripts/download_checkpoints.py                      # all released configs
uv run python scripts/download_checkpoints.py --configs default    # just one
```

This places them under `prism_monitor/checkpoints/<config>/`. The script then checks that every
file of the requested configs is present. If any are missing, it lists them and exits with
status 1; rerunning it fetches the rest.

## Inference on an (image, query) pair

```python
# run from prism_monitor/
from model import PRISMPredictor, load_image

prism = PRISMPredictor("default")   # checkpoints/default + its SigLIP2 encoder (GPU if available)
p_defer = prism.predict(
    [load_image("street_food.jpg")],
    ["Should I eat at this stall tonight?"],
)
print(f"P(needs user background) = {p_defer[0]:.3f}")   # defer if >= 0.5
```

`PRISMPredictor(checkpoint)` accepts a config name under `checkpoints/` or a path to a
checkpoint directory, and loads the encoder named in its `config.json`. `predict` takes
batches of images and queries. To use the model
without the encoder (for example on embeddings you computed yourself), call
`model.load_prism(checkpoint, device)`. It returns `(model, config)`, and the model is called as
`model(image_emb=..., text_emb=...)["safety_logits"]`.

**Preprocessing.** `predict` computes embeddings with the same `Encoder.embed()` path and image
preprocessing that produced the paper's embeddings: `load_image` downscales only images above 89M
pixels, each encoder's processor does the rest, and Qwen3-VL first resizes to 384x384 and then uses
its default dynamic resolution. Pillow 12 is required for paper-identical image preprocessing
(`pyproject.toml` pins `pillow>=12.0,<13`).

## Training

The training data (images plus question pairs with labels) is not released. To regenerate
it, run `dataset_building/generate_train_data.py`. It gathers a tone-balanced image pool from the
benchmark's source datasets (`benchmark/collection/source_data/`; see the benchmark README for
how to download them) and uses OpenAI VLMs to label image valence and generate matched safe and
unsafe question pairs, then writes image-level `dataset/splits/{train,val,test}.json`. Images
whose valence label or domain ID still fails after retries are excluded and listed, with the
failed step and error, in `dataset/train_data_failures.json`. A domain that still yields fewer
question pairs than requested after its retries is listed there too, with the requested and
produced counts (stage `pair_shortfall`, or `pair_generation` if it yielded none); the script
prints the requested, produced and missing pair totals. This needs `OPENAI_API_KEY`.

The script also fits a TF-IDF + logistic-regression text-only classifier and scores each
safe/unsafe pair by how confidently it separates the two questions,
`max |P(unsafe) − 0.5|` over the pair. A pair is kept if that score is at most
`--lexical_margin`. At the default margin of 0.5 (the setting used for the paper) every
score passes, so the filter removes no complete pair: it only drops a lone question whose
partner came back empty. The scores are then used only to pick the least
text-separable pairs as style examples for the balance-correction rounds. To drop
lexically easy pairs, pass a smaller margin, e.g. `--lexical_margin 0.3`.

```bash
cd prism_monitor/dataset_building
uv run python generate_train_data.py --max_images 600 --pairs_per_image 2   # paper settings
```

`--pairs_per_image 2` is also the default. The script refuses to start if any of its outputs
(`dataset/images/`, `dataset/train_data.json`, `dataset/train_data_failures.json`,
`dataset/splits/{train,val,test}.json`) already exists, unless `--overwrite` is passed.

Then cache the frozen encoder embeddings, train and evaluate:

```bash
cd prism_monitor
uv run python cache_embeddings.py                         # writes cache/<encoder>/{train,val,test}.pt
uv run python train_models.py                             # all four configs → checkpoints/<config>/
uv run python train_models.py --models default --wandb    # one config, log to W&B (WANDB_ENTITY env)
```

`bash run.sh` runs the three steps with `train_models.py --skip_existing`, so configs that already
have a `checkpoints/<config>/model.safetensors` (for example the released weights placed there by
`download_checkpoints.py`) are evaluated as they are instead of retrained; `bash run.sh --retrain`
passes `--overwrite` instead and retrains every config.

Training uses AdamW (lr 1e-4, weight decay 1e-5), batch size 32, up to 15 epochs with 20
warmup steps and cosine decay, and early stopping on val F1 (patience 5). The loss is
BCE on the deferral label plus 0.3 × conflict CE plus 0.1 × each tone CE, with seed 42
(`config.py`). The trainer saves the best epoch in the same `model.safetensors` +
`config.json` format as the released checkpoints, and writes the best val F1, epoch
counts, timing and per-epoch history to `training_info.json` next to them. On the paper's
splits and embedding cache, this code reproduces the released `default` checkpoint (best epoch 2,
val F1 0.91897).
`--checkpoint_dir` and `--cache_dir` override the default output and cache locations;
`cache_embeddings.py` also takes `--cache_dir` (and `--data_dir` for the split files).

Nothing is overwritten silently. `cache_embeddings.py` refuses to run if a selected encoder already has
cache files (a new Qwen3-VL cache also draws a new random projection, which would not match
checkpoints trained on the old one), and it stops with an error on an image it cannot read.
`train_models.py` refuses to train into a non-empty `checkpoints/<config>/`, such as one holding the
downloaded released weights; use another `--checkpoint_dir`, `--skip_existing` (skip configs that
already have a `model.safetensors`), or `--overwrite`. Both accept `--overwrite` to replace their
files; `train_models.py` also stops with an error if no epoch was saved.

## Evaluation

```bash
cd prism_monitor
uv run python evaluate.py                                       # all configs with a checkpoint
uv run python evaluate.py --models default evaclip_default \
    --checkpoint_dir checkpoints --cache_dir cache
```

This reports AUROC, AUPRC, F1 and accuracy at threshold 0.5, ECE, Brier score, and the
false-negative rate when deferring 30%/50% of queries, on the cached test split
(`cache/<encoder>/test.pt`, which you build with `cache_embeddings.py` from your own
splits). Results go to `results/eval_results.json`; an existing file is not replaced unless
`--overwrite` is passed (or use another `--output_dir`).

## Layout

| Path | Contents |
|---|---|
| `model/architecture.py` | `PRISMMonitor` head (Bi-FiLM fusion, modality gate, multi-task heads) |
| `model/encoders.py` | frozen SigLIP2 / CLIP / EVA-CLIP / Qwen3-VL encoders and preprocessing |
| `model/loading.py` | `load_prism`, `save_checkpoint`, `PRISMPredictor` |
| `model/trainer.py`, `model/dataset.py` | multi-task trainer, cached-embedding datasets |
| `config.py` | the four PRISM configs, training hyperparameters and default paths |
| `cache_embeddings.py`, `train_models.py`, `evaluate.py`, `run.sh` | pipeline entry points |
| `dataset_building/` | training-data generation |
