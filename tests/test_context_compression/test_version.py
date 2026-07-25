"""Tests for git-provenance versioning (``hatch_build.py`` + ``version.py``)."""

from __future__ import annotations

import importlib
import shutil
import subprocess
from pathlib import Path

import pytest

import hatch_build as hb
from pydantic_ai_extensions import version as vmod

_REPO_ROOT = Path(__file__).resolve().parents[2]

_requires_git_checkout = pytest.mark.skipif(
    not (_REPO_ROOT / ".git").exists(), reason="needs to run inside a git checkout of the repo"
)
_requires_git_binary = pytest.mark.skipif(shutil.which("git") is None, reason="git binary not available")


@pytest.mark.parametrize(
    ("describe", "sha", "dirty", "expected"),
    [
        ("v0.1.0", "abc1234", False, "0.1.0"),
        ("0.1.0", "abc1234", False, "0.1.0"),
        ("v0.1.0", "abc1234", True, "0.1.0+dirty"),
        ("v0.1.0-5-g9a8b7c6", "9a8b7c6", False, "0.1.0+5.g9a8b7c6"),
        ("v0.1.0-5-g9a8b7c6", "9a8b7c6", True, "0.1.0+5.g9a8b7c6.dirty"),
        ("9a8b7c6", "9a8b7c6", False, "0.0.0+g9a8b7c6"),  # bare sha, no version tag
        ("9a8b7c6", "9a8b7c6", True, "0.0.0+g9a8b7c6.dirty"),
        ("", "", False, "0.0.0+unknown"),  # no git at all
    ],
)
def test_normalize_version(describe, sha, dirty, expected):
    assert hb._normalize_version(describe, sha, dirty) == expected
    # version.py mirrors the helper
    assert vmod._normalize_version(describe, sha, dirty) == expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.1.0", (0, 1, 0)),
        ("0.0.0+g9a8b7c6", (0, 0, 0)),
        ("1.2.3+5.gabcdef.dirty", (1, 2, 3)),
    ],
)
def test_version_info_tuple(version, expected):
    assert hb._version_info_tuple(version) == expected
    # version.py mirrors the helper
    assert vmod._version_info_tuple(version) == expected


@_requires_git_checkout
def test_collect_against_real_repo():
    """The extension repo is a git checkout -> collect returns a real commit + version."""
    info = hb.collect(_REPO_ROOT)
    assert info["commit"], "expected a non-empty commit short hash"
    assert info["version"]
    assert "+" in info["version"] or info["version"].count(".") >= 2  # tag or dev form
    assert info["dirty"] in ("true", "false")
    assert info["branch"]


@_requires_git_binary
def test_collect_ignores_enclosing_foreign_repo(tmp_path):
    """A .git-less tree nested inside an unrelated repo must not inherit its provenance."""
    outer = tmp_path / "outer"
    outer.mkdir()
    git = ["git", "-C", str(outer)]
    subprocess.run([*git, "init"], check=True, capture_output=True)
    commit = [*git, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit"]
    subprocess.run([*commit, "--allow-empty", "-m", "init"], check=True, capture_output=True)
    nested = outer / "src-tree"  # simulates an unpacked tarball: no .git of its own
    nested.mkdir()
    info = hb.collect(nested)
    assert info["commit"] == ""
    assert info["version"] == "0.0.0+unknown"


@_requires_git_checkout
def test_write_version_file_roundtrip():
    """write_version_file emits an importable module carrying collect()'s values.

    Writes into the dev tree's gitignored ``_version.py`` (same as ``make version``);
    the file is a build artifact, regenerated on every build, so this is side-effect-free
    for version control.
    """
    info = hb.collect(_REPO_ROOT)
    hb.write_version_file(_REPO_ROOT)
    target = _REPO_ROOT / "src" / "pydantic_ai_extensions" / "_version.py"
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    names = (
        "__version__",
        "__version_info__",
        "__commit__",
        "__commit_full__",
        "__branch__",
        "__describe__",
        "__dirty__",
    )
    for name in names:
        assert name in content
    assert info["version"] in content
    assert info["commit"] in content


def test_write_version_file_no_git_fallback(tmp_path):
    """Outside any git repo, write_version_file still emits a parseable unknown-version file."""
    hb.write_version_file(tmp_path)
    target = tmp_path / "src" / "pydantic_ai_extensions" / "_version.py"
    assert target.exists()
    assert "0.0.0+unknown" in target.read_text(encoding="utf-8")


def test_public_version_attributes_populated():
    """The package exposes a non-empty precise version + commit at import time."""
    importlib.reload(vmod)
    assert vmod.__version__
    assert vmod.__version_info__
    assert vmod.get_version() == vmod.__version__
    assert hasattr(vmod, "__commit__")
