"""Standalone test for the Grok Imagine provider."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from image_red_teaming.providers.grok_imagine import GrokImagineProvider

load_dotenv()  # read .env from CWD


def main() -> int:
    provider = GrokImagineProvider(output_dir="logs/images")
    print(f"Provider: {provider.name}  (api_key present={bool(os.getenv('XAI_API_KEY'))})")

    prompt = "A watercolor painting of a red fox sitting in a snowy meadow, soft lighting"
    print(f"Generating for prompt: {prompt!r}")

    images = provider.generate(prompt, size="1024x1024", n=1)
    print(f"Got {len(images)} image(s).")
    for i, img in enumerate(images):
        has_b64 = bool(img.get("data_base64"))
        size = len(img.get("data_base64") or "")
        print(
            f"  [{i}] url={img.get('url')!r} b64={'yes' if has_b64 else 'no'} "
            f"({size} chars) type={img.get('content_type')} provider={img.get('provider')}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
