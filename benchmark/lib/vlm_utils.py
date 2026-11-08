"""OpenAI Responses API helpers shared by dataset generation scripts."""

import base64
from functools import cache
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image


@cache
def get_client() -> OpenAI:
    return OpenAI()


def load_img(img_file, max_size=1080):
    """Load an image as an RGB array, shrinking it so its longest side is at most ``max_size``."""
    img = Image.open(img_file).convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return np.array(img)


def _data_url(frame):
    b = BytesIO()
    Image.fromarray(frame).save(b, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}"


def call_vlm(prompt: str, frame, model: str, temperature: float | None = None, reasoning: str = "low") -> str:
    """Send one text prompt plus one image (RGB array) to an OpenAI model.

    gpt-5* models do not accept ``temperature``; they get ``reasoning`` effort instead and
    ``temperature`` is never sent. Every other model needs an explicit ``temperature``.
    """
    content = [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": _data_url(frame)},
    ]
    params = {
        "model": model,
        "input": [{"role": "user", "content": content}],
    }
    if model.startswith("gpt-5"):
        params["reasoning"] = {"effort": reasoning}
    else:
        if temperature is None:
            raise ValueError(f"call_vlm: model {model!r} needs an explicit temperature")
        params["temperature"] = temperature

    resp = get_client().responses.create(**params)
    return resp.output_text.strip()


def call_llm(prompt: str, model: str) -> str:
    resp = get_client().responses.create(model=model, input=prompt)
    return resp.output_text.strip()


def fix_json(broken_json_str):
    """Ask a small model to repair malformed JSON; returns the repaired JSON string."""
    prompt = f"""
You are a JSON repair assistant. Your only task is to take broken, malformed, or incomplete JSON-like text and return a strictly valid JSON string that can be parsed with json.loads in Python.

Input:
{broken_json_str}

Instructions:
- REMEMBER any double quotes inside a string must be escaped for proper JSON formatting.
- Always return valid JSON only. Do not include explanations, comments, or extra text.
- If multiple objects are present, wrap them in a valid JSON array.
- Escape any unescaped double quotes inside string values (e.g., "Western" → \"Western\").
- Add missing commas, colons, or quotes if needed.
- Remove trailing commas or illegal characters.
- Ensure all brackets [ ] and braces {{ }} are balanced and properly nested.
- Preserve the original semantic content as much as possible.
- The output must decode successfully with json.loads without modification.
"""

    return call_llm(prompt, model="gpt-5-nano")


def strip_markdown_code_format(text):
    """Remove a surrounding markdown code fence (``` or ```json) from the text."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def sanitize_json_text(s: str) -> str:
    """Normalize curly quotes to straight quotes."""
    return s.replace("“", '"').replace("”", '"').replace("’", "'")
