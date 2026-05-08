"""Multimodal preprocessing helpers (design §5.4).

What gets normalized for every image, before the model sees it:

1. **Mode → RGB.** RGBA gets composited onto white; grayscale gets
   broadcast to 3 channels; palette mode gets converted. The
   *"image preprocessor expects RGB but gets RGBA"* class of bug is
   impossible by construction.
2. **EXIF orientation** is honored (PIL's ``ImageOps.exif_transpose``).
3. **Sweet-spot downscale.** Images larger than the configured maximum
   pixel budget are downsized with high-quality resampling. Original
   dimensions are recorded in the manifest so the resize is auditable.
4. **Content hash.** Each pre-processed image is sha256-hashed (raw
   bytes) so the manifest can pin reproducible image inputs.

This module is in ``primitives/`` (a leaf in the import graph), so it
must not depend on the engine, models, tasks, manifest, or CaaS layers.
PIL is the only runtime dependency — already a hard dep per §16.8.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

# Qwen2.5-VL's documented sweet-spot patch granule (28×28 patches).
QWEN_VL_PATCH = 28
DEFAULT_MAX_PIXELS = 1280 * 768  # 983,040 — matches the §16.7 KB Qwen-VL fix
DEFAULT_MIN_PIXELS = 56 * 56  # the default vision-encoder minimum


@dataclass(frozen=True, slots=True)
class ProcessedImage:
    """An image after Anvil's normalization pass.

    Attributes:
        image: the normalized PIL.Image (RGB, EXIF-applied, downscaled).
        original_size: ``(width, height)`` before normalization.
        processed_size: ``(width, height)`` after normalization.
        hash: sha256 hex digest of the post-processed PNG bytes. Used by
            the manifest to pin image inputs reproducibly.
        pixel_count: ``processed_size[0] * processed_size[1]``.
    """

    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    hash: str
    pixel_count: int


def _to_image(src: Any) -> Image.Image:
    """Coerce a path / bytes / PIL.Image into a PIL.Image. Raises TypeError otherwise."""
    if isinstance(src, Image.Image):
        return src
    if isinstance(src, (str, Path)):
        return Image.open(str(src))
    if isinstance(src, bytes):
        return Image.open(io.BytesIO(src))
    raise TypeError(f"image input must be PIL.Image, path, or bytes; got {type(src).__name__}")


def _bucket_to_patch(value: int, granule: int) -> int:
    """Round ``value`` *down* to the nearest multiple of ``granule``.

    Anvil prefers down-rounding so the image always fits inside the
    configured budget. Vision encoders that expect a multiple-of-N
    geometry fail loudly when the image isn't bucketed.
    """
    if value <= 0:
        return granule
    return max(granule, (value // granule) * granule)


def preprocess_image(
    src: Any,
    *,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    patch_size: int = QWEN_VL_PATCH,
) -> ProcessedImage:
    """Normalize an image input. Returns a :class:`ProcessedImage`.

    Args:
        src: ``PIL.Image``, file path, or raw bytes.
        max_pixels: total-pixel budget; larger images are downscaled.
        min_pixels: lower bound; tiny images get upscaled.
        patch_size: granule for vision-encoder patch alignment (Qwen-VL = 28).

    Example:
        >>> from PIL import Image
        >>> img = Image.new("RGB", (4096, 2048))
        >>> p = preprocess_image(img, max_pixels=983040)
        >>> p.processed_size[0] * p.processed_size[1] <= 983040
        True
    """
    img = _to_image(src)

    # Honor EXIF orientation up-front so subsequent ops see the visual layout.
    img = ImageOps.exif_transpose(img) or img
    original = img.size

    # Mode normalization: anything → RGB.
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Sweet-spot rescale: keep aspect ratio; bucket to patch granule.
    width, height = img.size
    pixels = width * height
    if pixels > max_pixels or pixels < min_pixels:
        scale_target = max_pixels if pixels > max_pixels else min_pixels
        scale = (scale_target / pixels) ** 0.5
        new_w = _bucket_to_patch(int(width * scale), patch_size)
        new_h = _bucket_to_patch(int(height * scale), patch_size)
        # Keep within budget after bucketing (rounding can creep up).
        while new_w * new_h > max_pixels and (new_w > patch_size or new_h > patch_size):
            if new_w >= new_h and new_w > patch_size:
                new_w -= patch_size
            elif new_h > patch_size:
                new_h -= patch_size
            else:
                break
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    processed = img.size

    # Stable hash of the processed image's PNG-encoded bytes.
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False, compress_level=1)
    digest = "sha256:" + hashlib.sha256(buf.getvalue()).hexdigest()

    return ProcessedImage(
        image=img,
        original_size=original,
        processed_size=processed,
        hash=digest,
        pixel_count=processed[0] * processed[1],
    )


def is_multimodal_message(message: dict[str, Any]) -> bool:
    """True iff ``message['content']`` is a list of typed content parts.

    A plain text chat message has ``content: str``; a multimodal one has
    ``content: list[dict]`` where each entry carries ``type: 'image' | 'text' | …``.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and "type" in part for part in content)


def collect_images(messages: list[dict[str, Any]]) -> list[Any]:
    """Return every ``image`` payload found across ``messages``, in order.

    The payloads are returned **as-is** — paths/bytes/PIL — so the caller
    can pass them to :func:`preprocess_image` once.
    """
    out: list[Any] = []
    for msg in messages:
        if not is_multimodal_message(msg):
            continue
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                payload = part.get("image")
                if payload is not None:
                    out.append(payload)
    return out


def replace_images_with_processed(
    messages: list[dict[str, Any]],
    processed: list[ProcessedImage],
) -> list[dict[str, Any]]:
    """Return a copy of ``messages`` with each image payload replaced by its
    :class:`ProcessedImage.image`.

    Order is preserved: ``processed[i]`` corresponds to the i-th image
    yielded by :func:`collect_images`.
    """
    if not processed:
        return list(messages)
    iterator = iter(processed)
    out: list[dict[str, Any]] = []
    for msg in messages:
        if not is_multimodal_message(msg):
            out.append(msg)
            continue
        new_content: list[dict[str, Any]] = []
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("type") == "image":
                p = next(iterator)
                new_part = dict(part)
                new_part["image"] = p.image
                new_content.append(new_part)
            else:
                new_content.append(part)
        new_msg = dict(msg)
        new_msg["content"] = new_content
        out.append(new_msg)
    return out


__all__ = [
    "ProcessedImage",
    "preprocess_image",
    "is_multimodal_message",
    "collect_images",
    "replace_images_with_processed",
    "DEFAULT_MAX_PIXELS",
    "DEFAULT_MIN_PIXELS",
    "QWEN_VL_PATCH",
]
