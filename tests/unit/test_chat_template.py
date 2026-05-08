"""Tests for ``ChatTemplate`` (design §4.1)."""

from __future__ import annotations

import pytest

from anvil import ChatTemplate
from anvil.exceptions import TokenizationError


class TestChatTemplateHash:
    def test_hash_is_stable(self) -> None:
        a = ChatTemplate(jinja_source="{{ messages[0].content }}")
        b = ChatTemplate(jinja_source="{{ messages[0].content }}")
        assert a.hash == b.hash

    def test_hash_includes_fewshot_style(self) -> None:
        a = ChatTemplate(jinja_source="{{ x }}", fewshot_style="interleaved")
        b = ChatTemplate(jinja_source="{{ x }}", fewshot_style="concat-system")
        assert a.hash != b.hash

    def test_hash_starts_with_sha256_prefix(self) -> None:
        h = ChatTemplate(jinja_source="x").hash
        assert h.startswith("sha256:") and len(h) == len("sha256:") + 64


class TestChatTemplateCanonicalize:
    def test_whitespace_collapsed(self) -> None:
        a = ChatTemplate(jinja_source="{{ messages[0].content }}")
        b = ChatTemplate(jinja_source="{{    messages[0].content    }}")
        assert a.canonicalize() == b.canonicalize()
        assert a.hash == b.hash

    def test_comment_stripped(self) -> None:
        a = ChatTemplate(jinja_source="{# comment 1 #}{{ x }}")
        b = ChatTemplate(jinja_source="{# different comment #}{{ x }}")
        assert a.canonicalize() == b.canonicalize()
        assert a.hash == b.hash

    def test_quote_style_normalized(self) -> None:
        # Single vs double quotes inside a Jinja expression don't change semantics.
        a = ChatTemplate(jinja_source="{{ 'hello' }}")
        b = ChatTemplate(jinja_source='{{ "hello" }}')
        assert a.hash == b.hash

    def test_semantic_changes_visible_in_hash(self) -> None:
        a = ChatTemplate(jinja_source="{{ x }}")
        b = ChatTemplate(jinja_source="{{ y }}")
        assert a.hash != b.hash


class TestChatTemplateRender:
    def test_render_simple_message(self) -> None:
        ct = ChatTemplate(
            jinja_source="{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
        )
        out = ct.render([{"role": "user", "content": "hi"}], add_generation_prompt=False)
        assert "user: hi" in out

    def test_render_double_bos_raises(self) -> None:
        # If a template emits BOS twice (a real-world misconfiguration), we
        # surface it as a TokenizationError rather than silently letting the
        # double-BOS poison the run.
        ct = ChatTemplate(jinja_source="<s><s>{{ messages[0].content }}")
        with pytest.raises(TokenizationError, match="duplicate BOS"):
            ct.render([{"role": "user", "content": "hi"}])


class TestChatTemplateValidation:
    def test_invalid_fewshot_style_rejected(self) -> None:
        with pytest.raises(ValueError, match="fewshot_style"):
            ChatTemplate(jinja_source="x", fewshot_style="banana")  # type: ignore[arg-type]

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValueError, match="source"):
            ChatTemplate(jinja_source="x", source="invented")


def test_to_manifest_field_shape() -> None:
    ct = ChatTemplate(jinja_source="{{ x }}", name="trivial")
    field = ct.to_manifest_field()
    assert set(field) == {"hash", "version", "source", "fewshot_style"}
    assert field["fewshot_style"] == "interleaved"
    assert field["source"] == "user-supplied"
