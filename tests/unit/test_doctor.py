"""Tests for ``anvil doctor`` (design §16.10 M6, §16.12)."""

from __future__ import annotations

import json

from anvil.cli.doctor import (
    Check,
    overall_status,
    render_table,
    run_all_checks,
    to_json,
)


class TestCheckShape:
    def test_to_dict_round_trip(self) -> None:
        c = Check(name="x", status="ok", message="all good", detail={"k": "v"})
        d = c.to_dict()
        assert d == {
            "name": "x",
            "status": "ok",
            "message": "all good",
            "detail": {"k": "v"},
        }


class TestRenderTable:
    def test_renders_each_check(self) -> None:
        checks = [
            Check(name="a", status="ok", message="all good"),
            Check(name="bb", status="warn", message="not ideal"),
            Check(name="ccc", status="fail", message="broken"),
        ]
        table = render_table(checks)
        assert "a" in table and "ok" in table and "all good" in table
        assert "bb" in table and "warn" in table
        assert "ccc" in table and "fail" in table

    def test_empty_list(self) -> None:
        assert render_table([]) == "(no checks)"


class TestToJson:
    def test_round_trips_through_json(self) -> None:
        checks = [Check(name="x", status="ok", message="hi", detail={"a": 1})]
        text = to_json(checks)
        parsed = json.loads(text)
        assert parsed == [{"name": "x", "status": "ok", "message": "hi", "detail": {"a": 1}}]


class TestOverallStatus:
    def test_all_ok(self) -> None:
        assert overall_status([Check("a", "ok", "")]) == "ok"

    def test_warn_dominates_ok(self) -> None:
        checks = [Check("a", "ok", ""), Check("b", "warn", "")]
        assert overall_status(checks) == "warn"

    def test_fail_dominates_warn(self) -> None:
        checks = [Check("a", "warn", ""), Check("b", "fail", "")]
        assert overall_status(checks) == "fail"

    def test_empty_returns_ok(self) -> None:
        assert overall_status([]) == "ok"


class TestRunAllChecks:
    def test_runs_every_shipped_check(self) -> None:
        """The §16.10 M6 spec: doctor diagnoses 8/10 simulated environment problems.

        We can't simulate 10 distinct envs from inside one process, but we
        can confirm doctor runs every shipped check and produces valid
        output for the current env.
        """
        checks = run_all_checks()
        # The shipped check list has at least 8 entries (anvil, python, os,
        # torch, cuda, transformers, vllm, hf_token, hf_home_disk, caas_kb,
        # fast_paths). Confirms the §10.3 diagnosis surface is wired up.
        assert len(checks) >= 8
        # Every check produced a valid status.
        for c in checks:
            assert c.status in {"ok", "warn", "fail"}
            assert c.name and c.message

    def test_anvil_check_always_ok(self) -> None:
        from anvil.cli.doctor import _check_anvil_version

        c = _check_anvil_version()
        assert c.status == "ok"
        assert "anvil" in c.message.lower()

    def test_python_version_check(self) -> None:
        from anvil.cli.doctor import _check_python_version

        c = _check_python_version()
        # Either ok (3.11+) or fail (older). Either way, valid output.
        assert c.status in {"ok", "fail"}
