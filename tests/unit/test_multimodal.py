"""Tests for ``anvil.primitives.multimodal`` (design §5.4)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from anvil.primitives.multimodal import (
    DEFAULT_MAX_PIXELS,
    DEFAULT_MIN_PIXELS,
    QWEN_VL_PATCH,
    collect_images,
    is_multimodal_message,
    preprocess_image,
    replace_images_with_processed,
)


class TestPreprocessImage:
    def test_rgba_composite_to_rgb_white(self) -> None:
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))  # transparent red
        out = preprocess_image(img)
        assert out.image.mode == "RGB"
        # The transparent pixels should composite onto white (full alpha=0).
        assert out.image.getpixel((0, 0)) == (255, 255, 255)

    def test_grayscale_to_rgb(self) -> None:
        img = Image.new("L", (64, 64), 128)
        out = preprocess_image(img)
        assert out.image.mode == "RGB"

    def test_downscale_to_max_pixels(self) -> None:
        # 4K image — way over the default budget.
        img = Image.new("RGB", (3840, 2160))
        out = preprocess_image(img, max_pixels=DEFAULT_MAX_PIXELS)
        assert out.pixel_count <= DEFAULT_MAX_PIXELS
        assert out.original_size == (3840, 2160)

    def test_processed_size_aligned_to_patch(self) -> None:
        img = Image.new("RGB", (3840, 2160))
        out = preprocess_image(img, max_pixels=DEFAULT_MAX_PIXELS)
        assert out.processed_size[0] % QWEN_VL_PATCH == 0
        assert out.processed_size[1] % QWEN_VL_PATCH == 0

    def test_below_min_pixels_upscaled(self) -> None:
        img = Image.new("RGB", (16, 16))
        out = preprocess_image(img, min_pixels=DEFAULT_MIN_PIXELS)
        assert out.pixel_count >= DEFAULT_MIN_PIXELS

    def test_hash_is_deterministic(self) -> None:
        img1 = Image.new("RGB", (128, 64), (200, 100, 50))
        img2 = Image.new("RGB", (128, 64), (200, 100, 50))
        a = preprocess_image(img1)
        b = preprocess_image(img2)
        assert a.hash == b.hash
        assert a.hash.startswith("sha256:")

    def test_hash_changes_with_pixels(self) -> None:
        a = preprocess_image(Image.new("RGB", (128, 64), (200, 100, 50)))
        b = preprocess_image(Image.new("RGB", (128, 64), (210, 100, 50)))
        assert a.hash != b.hash

    def test_accepts_bytes(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64)).save(buf, format="PNG")
        out = preprocess_image(buf.getvalue())
        assert out.processed_size == (64, 64)

    def test_unsupported_input_raises(self) -> None:
        with pytest.raises(TypeError, match="image input must be"):
            preprocess_image(42)  # type: ignore[arg-type]


class TestMessageHelpers:
    def test_is_multimodal_message_true(self) -> None:
        msg = {
            "role": "user",
            "content": [{"type": "image", "image": "x"}, {"type": "text", "text": "hi"}],
        }
        assert is_multimodal_message(msg)

    def test_is_multimodal_message_false_for_plain_text(self) -> None:
        assert not is_multimodal_message({"role": "user", "content": "hi"})

    def test_collect_images_in_order(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "img-a"},
                    {"type": "text", "text": "what?"},
                    {"type": "image", "image": "img-b"},
                ],
            },
        ]
        assert collect_images(msgs) == ["img-a", "img-b"]

    def test_replace_images_with_processed_round_trip(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (32, 32))},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
        processed = [preprocess_image(msgs[0]["content"][0]["image"])]
        out = replace_images_with_processed(msgs, processed)
        # The image payload is replaced with the processed PIL.Image.
        assert out[0]["content"][0]["image"] is processed[0].image
        # Text part untouched.
        assert out[0]["content"][1] == {"type": "text", "text": "describe"}
