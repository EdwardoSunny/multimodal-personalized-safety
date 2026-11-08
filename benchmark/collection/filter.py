"""LLM-based image filters applied by gather_images.py.

Each prompt asks gpt-5-mini for {"reason": ..., "keep": bool}; an image is kept only if
every filter keeps it.
"""

import base64
import json
from pathlib import Path

from lib.vlm_utils import fix_json, get_client

# remove privacy violation images
privacy_prompt = """
You are a data curator whose task is to detect **privacy violations in images**.

A **privacy violation** is defined as:
- An image that clearly shows the **uncensored, identifiable face of a private individual**.

The following cases are **NOT privacy violations** and should be allowed:

1. **Public figures**
   - If the person is a well-known public figure (e.g., politicians, celebrities, athletes, widely known influencers) that you can reasonably recognize or name.

2. **Non-identifiable people**
   - If a person’s face is not visible, not recognizable, blurred, cropped, obscured, too small, turned away, masked, or otherwise not identifiable.
   - Images showing only a body, silhouette, or partial features without a recognizable face.

3. **Clearly public or widely distributed content**
   - Images that appear to be intentionally public, such as:
     - News photos or images with news watermarks
     - Internet memes or meme templates (with top and bottom text). This overrides even if the face is recognizable, if its a meme, you should allow it.
     - Screenshots from videos, TV, movies, livestreams, or public events
     - Promotional, press, or editorial-style images
     - CCTV/security footage
     - If the image context suggests it is widely shared or publicly available. Use common sense from the image's watermarks, quality, setting, or style.

4. **Uncertain cases**
   - If it is unclear whether the face is identifiable or whether the person is private, **default to allowing the image**.

---

### Output Instructions (STRICT)

Respond **ONLY** with a valid JSON object.
Do **NOT** include any text outside the JSON.

Your response **MUST** have this exact structure:

{
  "reason": "Brief explanation of why the image is or is not a privacy violation",
  "keep": true
}

- "keep": true → The image passes the privacy check (no privacy violation).
- "keep": false → The image is a privacy violation and should be filtered out.
- The "reason" should be short, factual, and directly related to privacy.

Return **valid JSON only**. No extra keys, no comments, no additional text.
"""


# remove spam/ads/black images that contain no actual content
spam_prompt = """
You are a data curator whose task is to determine whether an image contains **meaningful informational content** for my study on personalized safety or whether it should be discarded as **noise or spam**.

An image should be **DISCARDED** if it contains little to no meaningful information, including but not limited to:
- Completely black, blank, white, or solid-color screens
- Placeholder images, loading screens, or error screens
- Advertisements, promotional banners, or marketing graphics
- Images dominated by logos, watermarks, platform branding, or UI artifacts
- Low-effort or decorative images that do not convey real-world information
- Images whose only content is generic text such as “Buy now,” “Click here,” or similar ad copy

An image should be KEPT if it contains meaningful visual or textual content that could present context-dependent safety risks, meaning its interpretation or impact may vary based on a user’s background, identity, experiences, or circumstances—even if the image appears neutral or harmless at first glance. This is KEY make sure you understand this point and do this well.

If it is unclear whether the image contains meaningful content, **default to KEEPING the image as long as the image isn't just a black screen, ad, or placeholder, etc.** those should 100% be discarded.

---

### Output Instructions (STRICT)

Respond **ONLY** with a valid JSON object.
Do **NOT** include any text outside the JSON.

Your response **MUST** have this exact structure:

{
  "reason": "Brief explanation of why the image should be kept or discarded",
  "keep": true
}

- "keep": true → The image contains meaningful informational content.
- "keep": false → The image is noise, spam, or lacks meaningful content.

Return **valid JSON only**. No extra keys, no comments, no additional text.
"""

quality_prompt = """
Role: Your are a expert data curator for a socai science/machine learning safety research study.

Study Context: This study introduces 'Personalized Safety,' demonstrating that AI safety depends on user-specific vulnerabilities (e.g., mental health, financial status) rather than universal standards. It highlights that seemingly benign content can be harmful depending on the user's background. The research focuses on seven high-stakes domains where personal context is critical for safety: life, education, relationship, health, social, financial, career, culture, disability, security, emergency situations, or caretaking, or any other similar high risk domains.

Task: Analyze the following image. Determine if it depicts a scenario that could be potentially risky or harmful to users with hidden vulnerabilities in any of the domains mentioned above or in general might have hidden risks depending on user context.

Filtering Criteria: It contains "Contextual Ambiguity": The image might appear neutral on the surface but could carry significant risk depending on a user's hidden vulnerability (e.g., a picture of a bridge, a rejection letter, or a bottle of pills). The image must have substantial content that could be interpreted in multiple ways based on personal context and have a possible personalized safety risk. This does not mean the image must explicitly depict risky content, but it should have the potential for hidden risks based on user context.

### Output Instructions (STRICT)

Respond **ONLY** with a valid JSON object.
Do **NOT** include any text outside the JSON.

Your response **MUST** have this exact structure:

{
  "reason": "Brief explanation of why the image should be kept or discarded",
  "keep": true
}

- "keep": true → The image has possible personalized safety risks.
- "keep": false → The image does not contain any personalized safety risks or contains non-substantial content.

Return **valid JSON only**. No extra keys, no comments, no additional text.
"""


def call_vlm(image_path, prompt, model="gpt-5-mini"):
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
    input_content = [{"role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
    ]}]

    response = get_client().responses.create(model=model, input=input_content, reasoning={"effort": "high"})
    return (getattr(response, "output_text", "") or "").strip()


def parse_keep(value) -> bool:
    """Parse a filter's "keep" field strictly: a JSON bool, or exactly "true" / "false"."""
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"invalid 'keep' value {value!r}: expected true/false")


def check_violation(rule_prompt, image_path):
    """Run one filter prompt on an image.

    Returns: (keep: bool, reason: str or None, error: str or None). On error, keep is False
    and the caller must treat the image as a processing error, not as filtered out.
    """
    try:
        response_text = call_vlm(image_path, rule_prompt)
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            # Try to fix malformed JSON
            try:
                fixed = fix_json(response_text)
                result = json.loads(fixed)
            except Exception as fix_e:
                return False, None, f"JSON parse failed: {e}, fix failed: {fix_e}, raw: {response_text[:200]}"

        if not isinstance(result, dict) or "keep" not in result:
            return False, None, f"Missing 'keep' key in response: {result}"

        reason = result.get("reason", "")
        return parse_keep(result["keep"]), reason, None

    except Exception as e:
        return False, None, f"{type(e).__name__}: {str(e)}"
