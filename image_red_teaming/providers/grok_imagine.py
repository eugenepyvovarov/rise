from __future__ import annotations

import base64
import logging
import os
from math import gcd
from pathlib import Path
from typing import Any, Dict, List
from urllib.request import urlopen
from uuid import uuid4

import httpx

from image_red_teaming.providers.base import GeneratedImage, ImageGenProvider, GuardrailError

logger = logging.getLogger(__name__)

# Default xAI endpoint. xAI exposes an OpenAI-compatible images HTTP API at
# https://api.x.ai/v1, but its parameter set differs from OpenAI's (it uses
# `aspect_ratio` instead of pixel `size`), so we talk to it with a raw client
# rather than the strict OpenAI SDK client.
DEFAULT_BASE_URL = "https://api.x.ai/v1"

# Ratio (reduced) -> xAI aspect_ratio string. xAI does not accept pixel sizes,
# so a `size=` kwarg is translated to the nearest supported aspect ratio.
_SIZE_TO_ASPECT_RATIO = {
    (1, 1): "1:1",
    (2, 3): "2:3",
    (3, 2): "3:2",
    (3, 4): "3:4",
    (4, 3): "4:3",
    (9, 16): "9:16",
    (16, 9): "16:9",
    (2, 1): "2:1",
    (1, 2): "1:2",
}


class GrokImagineProvider(ImageGenProvider):
    """
    Minimal connector for xAI "Grok Imagine" image generation.

    Authentication:
      - Uses the `XAI_API_KEY` environment variable (xAI's default).
      - Optionally reads `XAI_BASE_URL` for a custom endpoint (defaults to
        https://api.x.ai/v1).

    xAI's images API speaks the OpenAI-compatible HTTP contract (POST
    /images/generations returns {"data": [{"url", "mime_type"}]}), but accepts a
    different set of parameters. It returns URLs + mime type rather than base64,
    so images are downloaded from the returned URLs.
    """

    name = "grok-imagine"

    def __init__(
        self,
        model_id: str | None = None,
        api_key_env: str = "XAI_API_KEY",
        base_url: str | None = None,
        output_dir: str | Path | None = None,
        default_params: Dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id or "grok-imagine-image-2.0"
        self.name = f"grok-imagine/{self.model_id}"
        self.output_dir = Path(output_dir) if output_dir else Path("logs/images")
        self.default_params: Dict[str, Any] = dict(default_params or {})
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} environment variable is not set. "
                "Set it before using the Grok Imagine provider."
            )
        base_url_env = os.getenv("XAI_BASE_URL", DEFAULT_BASE_URL)
        self.base_url = (base_url or base_url_env).rstrip("/")
        self.api_key = api_key

    def _is_guardrail_error(self, message: str) -> bool:
        lowered = message.lower()
        return any(
            token in lowered
            for token in (
                "content_policy",
                "safety",
                "moderation",
                "content-moderated",
                "rejected by content",
                "policy violation",
                "blocked",
            )
        )

    def _size_to_aspect_ratio(self, size: str) -> str | None:
        try:
            width, height = (int(x) for x in str(size).lower().split("x"))
        except (ValueError, AttributeError):
            return None
        if width <= 0 or height <= 0:
            return None
        divisor = gcd(width, height) or 1
        return _SIZE_TO_ASPECT_RATIO.get((width // divisor, height // divisor))

    def _normalise_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate OpenAI-style kwargs into what xAI accepts."""
        if "image_size" in kwargs and "size" not in kwargs:
            kwargs["size"] = kwargs.pop("image_size")
        # xAI rejects pixel `size`; translate to aspect_ratio, else drop it.
        if "size" in kwargs:
            raw_size = kwargs.pop("size")
            aspect = self._size_to_aspect_ratio(raw_size)
            if aspect:
                kwargs.setdefault("aspect_ratio", aspect)
            else:
                logger.warning(
                    "Unsupported pixel size %r for xAI; ignored (use aspect_ratio)",
                    raw_size,
                )
        return kwargs

    def _persist_image_bytes(self, image_bytes: bytes) -> tuple[str | None, str | None, str | None]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            filename = f"{self.model_id.replace('/', '_')}_{uuid4().hex}.png"
            path = self.output_dir / filename
            try:
                with path.open("xb") as f:
                    f.write(image_bytes)
                resolved = path.resolve()
                file_url = resolved.as_uri()
                vscode_url = f"vscode://file{resolved.as_posix()}"
                return str(resolved), file_url, vscode_url
            except FileExistsError as exc:
                logger.warning("Image filename collision at %s: %s", path, exc)
                continue
        logger.warning("Failed to allocate unique image filename after retries")
        return None, None, None

    def generate(self, prompt: str, **kwargs: Any) -> List[GeneratedImage]:
        """
        Submit a request to xAI image generations and return a normalized list.

        Extra keyword arguments are passed into the request body, e.g.:
          - n=2
          - aspect_ratio="1:1"  (a `size="1024x1024"` is translated to this)
        """
        body: Dict[str, Any] = {"model": self.model_id, "prompt": prompt}
        body.update(self.default_params)
        body.update(self._normalise_kwargs(dict(kwargs)))

        try:
            response = httpx.post(
                f"{self.base_url}/images/generations",
                json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            logger.warning(
                "xAI images/generations returned HTTP %s: %s",
                exc.response.status_code,
                detail[:500],
            )
            if self._is_guardrail_error(detail):
                raise GuardrailError(self.name, detail) from exc
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("xAI image generation failed")
            raise

        normalized: List[GeneratedImage] = []
        for item in (payload.get("data") or []):
            if isinstance(item, dict):
                raw_item: Any = dict(item)
            else:
                raw_item = getattr(item, "model_dump", lambda: dict(item.__dict__))()

            url = raw_item.get("url")
            if not url:
                continue
            b64: str | None = None
            local_path: str | None = None
            file_url: str | None = None
            vscode_url: str | None = None
            content_type = raw_item.get("mime_type") or "image/jpeg"

            # Verify TLS by default. Set XAI_INSECURE_DOWNLOAD=1 only if your
            # environment cannot validate xAI's CDN certificate.
            insecure = os.getenv("XAI_INSECURE_DOWNLOAD", "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            try:
                with httpx.Client(verify=not insecure) as client:
                    resp = client.get(url, timeout=60)
                    resp.raise_for_status()
                    raw_bytes = resp.content
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                local_path, file_url, vscode_url = self._persist_image_bytes(raw_bytes)
            except Exception:
                logger.exception("Failed to download or encode image from xAI URL", extra={"url": url})
                b64 = None

            record: GeneratedImage = GeneratedImage(
                url=local_path or url or "",
                data_base64=b64,
                content_type=content_type,
                provider=self.name,
                extra={
                    "raw": raw_item,
                    "local_path": local_path,
                    "file_url": file_url,
                    "vscode_url": vscode_url,
                },
            )
            normalized.append(record)
        return normalized
