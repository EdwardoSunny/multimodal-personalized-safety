"""Generate benchmark responses from an open-weight VLM served by vLLM (OpenAI-compatible API).

Start the server first, e.g.:
    vllm serve unsloth/gemma-3-4b-it-bnb-4bit --port 8000

Then (from benchmark/):
    python evaluation/generate_responses_hosted.py --model unsloth/gemma-3-4b-it-bnb-4bit --host localhost --port 8000

--model must match the name the server was started with. Writes
results/responses/<model with '/' -> '_'>_responses.json. An existing file is resumed (its
records are kept, missing images answered, failed responses retried); --overwrite starts a
fresh one.
"""

import argparse
import base64
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.response_generation import (
    EmptyResponseError,
    add_common_args,
    generate_responses,
    load_image_for_request,
)

MAX_TOKENS = 1024


def make_call_fn(client: OpenAI, model: str):
    """Return call_fn(prompt, image_path) -> response text for a vLLM chat-completions server.

    call_fn raises on API errors (the driver retries transient ones) and raises
    EmptyResponseError when the model returns no text.
    """

    def call_vllm(prompt: str, image_path) -> str:
        image_bytes, mime = load_image_for_request(image_path)
        b64 = base64.b64encode(image_bytes).decode()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise EmptyResponseError(f"empty message content (finish_reason={resp.choices[0].finish_reason})")
        return text

    return call_vllm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_model="unsloth/gemma-3-4b-it-bnb-4bit", default_workers=8)
    parser.add_argument("--host", default="localhost", help="vLLM server host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port (default: %(default)s)")
    args = parser.parse_args()

    client = OpenAI(base_url=f"http://{args.host}:{args.port}/v1", api_key="unused")
    generate_responses(
        make_call_fn(client, args.model),
        model=args.model,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
