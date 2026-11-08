"""Generate benchmark responses from an open-weight VLM served by vLLM on Modal.

Deploys evaluation/modal_vllm_server.py as a uniquely named Modal app configured for
--model/--gpu/--n-gpu/--max-model-len, waits until it is healthy, generates responses
through its OpenAI-compatible endpoint, and always stops the app afterwards. The endpoint URL
is public while the app runs; its /v1 API requires a random per-run API key that only this
process knows.

Usage (from benchmark/):
    python evaluation/generate_responses_modal.py --model google/gemma-3-12b-it --gpu H100 --n-gpu 1

Requires an authenticated Modal CLI (`modal setup`) and a Modal secret named
"huggingface-secret" containing HF_TOKEN. Writes
results/responses/<model with '/' -> '_'>_responses.json. An existing file is resumed (its
records are kept, missing images answered, failed responses retried); --overwrite starts a
fresh one.
"""

import argparse
import json
import re
import secrets
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.generate_responses_hosted import make_call_fn
from lib.response_generation import add_common_args, generate_responses

MODAL_SERVER_SCRIPT = Path(__file__).resolve().parent / "modal_vllm_server.py"
MODAL_APP_NAME_BASE = "vllm-inference-server"

HEALTH_CHECK_TIMEOUT = 600  # seconds to wait for the server to come up
HEALTH_CHECK_INTERVAL = 10  # seconds between health checks


def render_server_script(app_name: str, model: str, gpu: str, n_gpu: int, max_model_len: int, api_key: str) -> str:
    """Return the source of modal_vllm_server.py with its CONFIGURATION block filled in."""
    hf_overrides = None
    if "deepseek" in model.lower():
        # DeepSeek-VL2 models need --hf-overrides '{"architectures":["DeepseekVLV2ForCausalLM"]}'
        hf_overrides = json.dumps({"architectures": ["DeepseekVLV2ForCausalLM"]})

    values = {
        "APP_NAME": app_name,
        "MODEL_NAME": model,
        "GPU_TYPE": gpu,
        "N_GPU": n_gpu,
        "MAX_MODEL_LEN": max_model_len,
        "HF_OVERRIDES": hf_overrides,
        "API_KEY": api_key,
    }
    content = MODAL_SERVER_SCRIPT.read_text()
    for name, value in values.items():
        content, n = re.subn(rf"^{name} = .*$", f"{name} = {value!r}", content, count=1, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f"{MODAL_SERVER_SCRIPT} has no top-level `{name} = ...` assignment")
    return content


def deploy_modal_server(server_source: str, app_name: str) -> str:
    """Deploy the rendered server script and return its endpoint URL.

    If the deploy succeeds but no URL can be found in its output, the app is stopped before
    raising, so it does not keep running (and billing) unattended.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / MODAL_SERVER_SCRIPT.name
        script_path.write_text(server_source)
        result = subprocess.run(
            [sys.executable, "-m", "modal", "deploy", str(script_path)],
            capture_output=True,
            text=True,
        )

    output = result.stdout + result.stderr
    print(output)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to deploy Modal server: {output}")

    # e.g. https://<workspace>--vllm-inference-server-abc12345-serve.modal.run
    match = re.search(r"(https://[a-zA-Z0-9-]+--[a-zA-Z0-9-]+-serve\.modal\.run)", output)
    if not match:
        stop_modal_server(app_name)
        raise RuntimeError(f"Could not find server URL in deploy output: {output}")
    url = match.group(1)
    print(f"\nModal server deployed at: {url}")
    return url


def wait_for_server_healthy(url: str, timeout: int = HEALTH_CHECK_TIMEOUT):
    print("\nWaiting for server to be healthy...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if requests.get(f"{url}/health", timeout=30).status_code == 200:
                print("Server is healthy.")
                return
        except requests.exceptions.RequestException:
            pass
        print(f"  Server not ready yet... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(HEALTH_CHECK_INTERVAL)
    raise RuntimeError(f"Server did not become healthy within {timeout} seconds")


def stop_modal_server(app_name: str):
    print(f"\nStopping Modal app {app_name}...")
    result = subprocess.run(
        [sys.executable, "-m", "modal", "app", "stop", app_name], capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    print(output)
    if result.returncode == 0:
        print("Modal server stopped.")
    else:
        print(f"Warning: could not stop Modal app {app_name}; stop it manually: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_model="google/gemma-3-12b-it", default_workers=32)
    parser.add_argument("--gpu", default="H100", help="Modal GPU type, e.g. H100, A100, L4 (default: %(default)s)")
    parser.add_argument("--n-gpu", type=int, default=1, help="GPUs per server (tensor parallel size)")
    parser.add_argument(
        "--max-model-len", type=int, default=32768, help="vLLM max context length (lower it on KV-cache OOM)"
    )
    args = parser.parse_args()

    # Unique app name so concurrent runs do not clobber each other's deployments
    app_name = f"{MODAL_APP_NAME_BASE}-{uuid.uuid4().hex[:8]}"
    print("=" * 80)
    print(f"Deploying Modal app {app_name}: {args.model} on {args.n_gpu}x {args.gpu}, max_model_len={args.max_model_len}")
    print("=" * 80)

    # The deployed URL is public; vLLM only answers /v1 requests carrying this per-run key
    api_key = secrets.token_urlsafe(32)
    server_source = render_server_script(app_name, args.model, args.gpu, args.n_gpu, args.max_model_len, api_key)
    endpoint_url = deploy_modal_server(server_source, app_name)
    try:
        wait_for_server_healthy(endpoint_url)
        client = OpenAI(base_url=f"{endpoint_url}/v1", api_key=api_key)
        generate_responses(
            make_call_fn(client, args.model),
            model=args.model,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    finally:
        stop_modal_server(app_name)


if __name__ == "__main__":
    main()
