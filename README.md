<h1 align="center">When Seeing Overrides Knowing</h1>
<h3 align="center">Visual Dominance and Deferral-Based Method for Personalized Safety in VLMs</h3>

<p align="center">
  <a href="https://huggingface.co/datasets/EdwardoSunny/multimodal-personalized-safety"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-MPS--Bench-yellow" alt="Dataset"></a>
  <a href="https://huggingface.co/EdwardoSunny/multimodal-personalized-safety-prism"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-PRISM-orange" alt="Model"></a>
  <img src="https://img.shields.io/badge/COLM-2026-blue" alt="COLM 2026">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

<p align="center"><img src="assets/teaser.png" width="600" alt="Personalized safety failures in VLMs"></p>

Code, data, and checkpoints for the COLM 2026 paper **When Seeing Overrides Knowing: Visual Dominance and Deferral-Based Method for Personalized Safety in VLMs**.

A response that is reasonable in general can be unsafe for a particular user whose medical, emotional, or situational context the model doesn't know. This repository provides:

- **MPS-Bench**: 5,181 scenarios built from 584 real-world images across 12 high-risk domains. Each scenario pairs a query with a hidden high-risk user profile. Models answer with and without the profile, and an LLM judge scores both answers on five personalized-safety dimensions. Code: [`benchmark/`](benchmark/README.md).
- **PRISM**: a lightweight input monitor that predicts, from frozen image and text embeddings fused by bidirectional cross-modal modulation, when a query should be deferred for missing user context. Code: [`prism_monitor/`](prism_monitor/README.md).

<p align="center"><img src="assets/mps_bench_pipeline.png" width="100%" alt="MPS-Bench construction and evaluation pipeline"></p>

## Results

Frontier VLMs almost always answer directly instead of asking for missing user context:

<p align="center"><img src="assets/asking_behavior.png" width="600" alt="Asking behavior across VLMs"></p>

Safety scores across the 12 domains, without (generic) and with (personalized) the user's background:

<p align="center"><img src="assets/domain_safety_radar.png" width="100%" alt="Safety scores by domain"></p>

## Artifacts

| Artifact | Location |
|---|---|
| Benchmark data (gated) | [`EdwardoSunny/multimodal-personalized-safety`](https://huggingface.co/datasets/EdwardoSunny/multimodal-personalized-safety) |
| PRISM checkpoints | [`EdwardoSunny/multimodal-personalized-safety-prism`](https://huggingface.co/EdwardoSunny/multimodal-personalized-safety-prism) |

> ⚠️ **Content warning.** The benchmark includes images of weapons, injuries, disasters, and hateful memes, and scenarios involving self-harm risk and substance use.

## Setup

```bash
git clone https://github.com/EdwardoSunny/multimodal-personalized-safety
cd multimodal-personalized-safety
uv sync                      # Python 3.11–3.13
```

Accept the dataset terms on its Hugging Face page, authenticate (`hf auth login` or `HF_TOKEN`), then:

```bash
uv run python scripts/download_data.py          # -> benchmark/data/
uv run python scripts/download_checkpoints.py   # -> prism_monitor/checkpoints/
```

Both scripts check that every file arrived. If any are missing, they list them and exit with an error; rerunning fetches the rest.

## Quick start

Evaluate a model on MPS-Bench and judge its responses. This needs `OPENAI_API_KEY`.

```bash
cd benchmark
uv run python evaluation/generate_responses_api.py --model gpt-5-mini
uv run python evaluation/judge.py --response-file results/responses/gpt-5-mini_responses.json
uv run python evaluation/classify_asking_behavior.py --models gpt-5-mini
uv run python evaluation/summarize.py --models gpt-5-mini     # safety scores and % direct, as in the paper
```

Open-weight models can be run through any OpenAI-compatible vLLM server (`generate_responses_hosted.py`) or on Modal (`generate_responses_modal.py`). [`benchmark/README.md`](benchmark/README.md) covers the data format, the judge rubric, the exact models evaluated in the paper, how failed responses are handled, how to build new benchmark data from raw image sources, and the changes from the paper code.

Score an (image, query) pair with PRISM:

```python
# run from prism_monitor/
from model import PRISMPredictor, load_image

prism = PRISMPredictor("default")
print(prism.predict([load_image("photo.jpg")], ["Should I try this tonight?"])[0])  # defer if >= 0.5
```

[`prism_monitor/README.md`](prism_monitor/README.md) covers the three released models, training (including the Qwen3-VL model), generating training data, and evaluation.

**Remote code.** The Modal vLLM server starts vLLM with `--trust-remote-code`, and the EVA-CLIP PRISM encoder loads `BAAI/EVA-CLIP-8B` with `trust_remote_code=True` at a pinned commit.

## License

Code and PRISM checkpoints are released under the [MIT License](LICENSE). The benchmark annotations are released under CC BY-NC 4.0. Images remain under the terms of their original sources and are distributed for non-commercial research only (see the dataset card).

To request removal of an image (for example, if you are depicted in it or hold rights to it), email edwardsun12895@g.ucla.edu with the image filename.

## Citation

```bibtex
@inproceedings{sun2026seeing,
  title     = {When Seeing Overrides Knowing: Visual Dominance and Deferral-Based Method for Personalized Safety in {VLMs}},
  author    = {Sun, Edward and Wu, Yuchen and Ma, Zixian and Jiang, Eric Hanchen and Xiao, Yijia and Yi, Xiaoyuan and Krishna, Ranjay and Wang, Wei and Wang, Jindong and Caliskan, Aylin},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```
