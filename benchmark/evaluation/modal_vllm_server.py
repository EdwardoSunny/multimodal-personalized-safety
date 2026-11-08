"""
Modal vLLM inference server (OpenAI-compatible) used by generate_responses_modal.py.

generate_responses_modal.py does not deploy this file directly: it renders a per-run copy
with the CONFIGURATION values below replaced (model, GPU, context length, unique app
name), deploys that copy, and stops the app when generation finishes. The values checked
in here are only the defaults used when deploying this file by hand:

    modal deploy evaluation/modal_vllm_server.py

Requires a Modal secret named "huggingface-secret" containing HF_TOKEN.

The endpoint URL is public. generate_responses_modal.py sets API_KEY to a random per-run token,
so vLLM rejects /v1 requests without it (`/health` stays open). Deployed by hand with
API_KEY = None, the server answers anyone who has the URL while it runs.
"""

import modal

# =============================================================================
# CONFIGURATION - rewritten per run by generate_responses_modal.py
# =============================================================================

APP_NAME = "vllm-inference-server"
MODEL_NAME = "google/gemma-3-12b-it"
GPU_TYPE = "H100"
N_GPU = 1
MAX_MODEL_LEN = 32768
# Optional JSON string passed to vLLM via --hf-overrides,
# e.g. '{"architectures":["DeepseekVLV2ForCausalLM"]}'
HF_OVERRIDES = None
# Bearer token vLLM requires on /v1 requests (--api-key); None leaves the endpoint unauthenticated
API_KEY = None

# =============================================================================

# Fast boot mode: True = faster startup (--enforce-eager), False = better inference performance
FAST_BOOT = True

MINUTES = 60  # seconds
SCALEDOWN_WINDOW = 15 * MINUTES
STARTUP_TIMEOUT = 10 * MINUTES
MAX_CONCURRENT_INPUTS = 32

VLLM_PORT = 8000

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11",
        "bitsandbytes>=0.48.2",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Volumes cache model weights and vLLM compilation artifacts across runs
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=STARTUP_TIMEOUT,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def serve():
    """Spawn a vLLM server listening at VLLM_PORT."""
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
    ]

    if MAX_MODEL_LEN is not None:
        cmd += ["--max-model-len", str(MAX_MODEL_LEN)]

    if HF_OVERRIDES is not None:
        cmd += ["--hf-overrides", HF_OVERRIDES]

    if API_KEY is not None:
        cmd += ["--api-key", API_KEY]

    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print("Starting vLLM server with command:")
    print(" ".join("<api key>" if API_KEY is not None and part == API_KEY else part for part in cmd))

    # Run without shell so JSON quoting is preserved
    subprocess.Popen(cmd)
