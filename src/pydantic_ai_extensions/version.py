"""Package version + git build provenance.

The authoritative values are baked into ``_version.py`` at build time by
``hatch_build.py`` (a hatchling build hook, no third-party dependency). When
``_version.py`` is absent -- e.g. importing directly from a fresh source checkout
before any build -- we fall back to ``git describe`` at import time so the version is
still meaningful on any branch and still carries the commit short hash.

For an installed wheel or editable install, ``_version.py`` is always present.

Public attributes:
- ``__version__``      precise version, e.g. ``1.2.3`` / ``1.2.3+5.gabcdef.dirty``
- ``__version_info__`` tuple form of the release segment, e.g. ``(1, 2, 3)``
- ``__commit__``       short commit hash (e.g. ``abcdef``), or ``""`` if unknown
- ``__commit_full__``  full commit hash, or ``""``
- ``__branch__``       branch name, or ``(detached)`` / ``(unknown)``
- ``__describe__``     raw ``git describe`` output
- ``__dirty__``        ``"true"`` / ``"false"``
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

__all__ = [
    "__branch__",
    "__commit__",
    "__commit_full__",
    "__describe__",
    "__dirty__",
    "__version__",
    "__version_info__",
    "get_version",
]

_UNKNOWN_VERSION = "0.0.0+unknown"
_REPO_ROOT = Path(__file__).resolve().parents[2]  # src/<pkg>/version.py -> repo root


def _normalize_version(describe: str, short_sha: str, dirty: bool) -> str:
    """Turn stripped ``git describe`` output into a PEP 440-ish version string.

    Mirrors ``hatch_build._normalize_version`` -- keep the two in sync.
    """
    m = re.match(r"^v?(\d[^-]*)-(\d+)-g([0-9a-f]+)$", describe)
    if m:
        base, local = m.group(1), f"{m.group(2)}.g{m.group(3)}"
    elif describe and re.fullmatch(r"[0-9a-f]{7,40}", describe):
        # `--always` fallback: bare abbreviated commit, no version tag reachable
        base, local = "0.0.0", f"g{describe}"
    else:
        tag = re.sub(r"^v", "", describe)
        if tag and re.match(r"^\d", tag):
            # a clean version tag, e.g. v1.2.3 -> 1.2.3
            base, local = tag, ""
        else:  # no git available at all
            base, local = "0.0.0", "unknown"
    if dirty:
        local = f"{local}.dirty" if local else "dirty"
    return f"{base}+{local}" if local else base


def _version_info_tuple(version: str) -> tuple[int | str, ...]:
    base = version.split("+", 1)[0]
    parts: list[int | str] = []
    for chunk in base.split("."):
        if chunk.isdigit():
            parts.append(int(chunk))
        elif chunk:
            parts.append(chunk)
    return tuple(parts)


def _from_git() -> dict[str, str]:
    """Best-effort runtime derivation when ``_version.py`` is absent (fresh checkout)."""

    def run(*args: str) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", str(_REPO_ROOT), *args],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return r.stdout.strip() if r.returncode == 0 else ""

    def unknown() -> dict[str, str]:
        return {
            "version": _UNKNOWN_VERSION,
            "commit": "",
            "commit_full": "",
            "branch": "(unknown)",
            "describe": "(unknown)",
            "dirty": "false",
        }

    # Only trust git when _REPO_ROOT is the top level of its own work tree: a
    # .git-less source tree nested inside an unrelated repository must not
    # silently inherit that repo's tags and commits.
    top = run("rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != _REPO_ROOT.resolve():
        return unknown()

    describe_raw = run("describe", "--tags", "--always", "--match=v[0-9]*", "--dirty")
    dirty = bool(describe_raw) and describe_raw.endswith("-dirty")
    describe = describe_raw[: -len("-dirty")] if dirty else describe_raw
    short = run("rev-parse", "--short", "HEAD")
    full = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    return {
        "version": _normalize_version(describe, short, dirty) or _UNKNOWN_VERSION,
        "commit": short,
        "commit_full": full,
        "branch": branch or "(detached)",
        "describe": describe or "(unknown)",
        "dirty": "true" if dirty else "false",
    }


try:
    from ._version import (  # type: ignore[attr-defined]
        __branch__,
        __commit__,
        __commit_full__,
        __describe__,
        __dirty__,
        __version__,
        __version_info__,
    )
except ImportError:
    _info = _from_git()
    __version__: str = _info["version"] or _UNKNOWN_VERSION
    __version_info__: tuple[int | str, ...] = _version_info_tuple(__version__)
    __commit__: str = _info["commit"]
    __commit_full__: str = _info["commit_full"]
    __branch__: str = _info["branch"]
    __describe__: str = _info["describe"]
    __dirty__: str = _info["dirty"]


def get_version() -> str:
    """Return the precise version string (tag + commit + dirty)."""
    return __version__
