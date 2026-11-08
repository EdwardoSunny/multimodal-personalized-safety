"""Generate benchmark responses from a hosted API model (OpenAI GPT-* or Google Gemini).

Usage (from benchmark/):
    python evaluation/generate_responses_api.py --model gpt-5-mini
    python evaluation/generate_responses_api.py --model gemini-2.5-flash --limit 1

Requires OPENAI_API_KEY (gpt-*) or GEMINI_API_KEY / GOOGLE_API_KEY (gemini-*).
Writes results/responses/<model>_responses.json. An existing file is resumed (its records are
kept, missing images answered, failed responses retried); --overwrite starts a fresh one.
"""

import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.response_generation import (
    EmptyResponseError,
    add_common_args,
    generate_responses,
    load_image_for_request,
)


def make_call_fn(model: str):
    """Return call_fn(prompt, image_path) -> response text for an OpenAI or Gemini model.

    call_fn raises on API errors (the driver retries transient ones) and raises
    EmptyResponseError when the model returns no text.
    """
    if "gpt" in model:
        from openai import OpenAI

        client = OpenAI()

        def call_openai(prompt: str, image_path) -> str:
            image_bytes, mime = load_image_for_request(image_path)
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            if "gpt-5" in model:
                input_content = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                        ],
                    }
                ]
                r = client.responses.create(
                    model=model, input=input_content, reasoning={"effort": "minimal"}
                )
                text = (getattr(r, "output_text", "") or "").strip()
                if not text:
                    raise EmptyResponseError(f"empty output_text (status={r.status}, incomplete={r.incomplete_details})")
                return text
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ]
            r = client.chat.completions.create(model=model, messages=msgs)
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise EmptyResponseError(f"empty message content (finish_reason={r.choices[0].finish_reason})")
            return text

        return call_openai

    if model.startswith("gemini"):
        from google import genai

        client = genai.Client()

        def call_gemini(prompt: str, image_path) -> str:
            image_bytes, mime = load_image_for_request(image_path)
            parts = [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": base64.b64encode(image_bytes).decode("utf-8"),
                    }
                },
            ]
            resp = client.models.generate_content(
                model=model, contents=[{"role": "user", "parts": parts}]
            )
            text = (getattr(resp, "text", "") or "").strip()
            if not text:
                finish = [c.finish_reason for c in (resp.candidates or [])]
                raise EmptyResponseError(f"empty text (finish_reasons={finish}, prompt_feedback={resp.prompt_feedback})")
            return text

        return call_gemini

    raise ValueError(f"Unsupported API model {model!r}: expected an OpenAI gpt-* or Google gemini-* model")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_model="gpt-5-mini", default_workers=8)
    args = parser.parse_args()

    generate_responses(
        make_call_fn(args.model),
        model=args.model,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
