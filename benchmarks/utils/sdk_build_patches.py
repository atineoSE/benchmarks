"""
Host-side patches for the bundled software-agent-sdk.

1) ``_extract_tarball`` race fix: pass ``path=dest`` directly to ``extractall``
   instead of using ``contextlib.chdir(dest)``. ``chdir`` mutates the
   process-wide cwd, so concurrent workers building docker contexts in
   parallel race each other: one worker's extraction writes into another's
   tmp dir, and cleanup of one tmp dir can leave another worker holding a
   dangling cwd that breaks the next ``os.getcwd()`` (typically inside
   ``Path.absolute`` during pydantic validation).

2) Shared SDK sdist: ``DockerDevWorkspace._build_image_from_base`` used to
   construct ``BuildOptions`` without ``prebuilt_sdist``, so every worker
   that hit the on-the-fly build path ran its own ``uv build --sdist``
   against the shared ``vendor/software-agent-sdk`` source tree. The builds
   raced over the workspace's ``UNKNOWN.egg-info`` and ``unknown-0.0.0/``
   staging dirs (the workspace pyproject has no ``[project]`` table, so
   setuptools falls back to UNKNOWN/0.0.0), produced megabytes of
   duplicated "copying ..." output per worker and never converged.
   We now build the sdist once per process under a lock and inject the
   resulting tarball as ``prebuilt_sdist`` so the SDK build pipeline skips
   the per-image ``uv build``.
"""

from __future__ import annotations

import atexit
import shutil
import tarfile
import threading
from pathlib import Path


_sdist_lock = threading.Lock()
_sdist_path: Path | None = None
_sdist_temp_root: Path | None = None


def _safe_extract_tarball(tarball: Path, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        for m in tar.getmembers():
            name = m.name.lstrip("./")
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise RuntimeError(f"Unsafe path in sdist: {m.name}")
        tar.extractall(path=str(dest), filter="data")


def _cleanup_shared_sdist() -> None:
    global _sdist_path, _sdist_temp_root
    if _sdist_temp_root is not None:
        shutil.rmtree(_sdist_temp_root, ignore_errors=True)
        _sdist_temp_root = None
        _sdist_path = None


def _get_or_build_shared_sdist() -> Path:
    """Build the SDK sdist once per process; subsequent callers reuse it."""
    global _sdist_path, _sdist_temp_root
    if _sdist_path is not None:
        return _sdist_path
    with _sdist_lock:
        if _sdist_path is not None:
            return _sdist_path
        # Deferred import: this runs only after DockerDevWorkspace is invoked,
        # well after sitecustomize-time module import.
        from benchmarks.utils.build_utils import _pre_build_sdist

        sdist = _pre_build_sdist()
        _sdist_temp_root = sdist.parent
        _sdist_path = sdist
        atexit.register(_cleanup_shared_sdist)
        return _sdist_path


def _patched_build_image_from_base(*, base_image, target, platform) -> str:
    from openhands.agent_server.docker.build import BuildOptions, build

    if "ghcr.io/openhands/agent-server" in base_image:
        raise RuntimeError(
            "base_image cannot be a pre-built agent-server image. "
            "Use server_image=... instead."
        )

    build_opts = BuildOptions(
        base_image=base_image,
        target=target,
        platforms=[platform],
        push=False,
        prebuilt_sdist=_get_or_build_shared_sdist(),
    )
    tags = build(opts=build_opts)
    if not tags:
        raise RuntimeError("Build failed, no image tags returned")
    return tags[0]


def apply_host_patches() -> None:
    from openhands.agent_server.docker import build

    build._extract_tarball = _safe_extract_tarball

    from openhands.workspace.docker.dev_workspace import DockerDevWorkspace

    DockerDevWorkspace._build_image_from_base = staticmethod(
        _patched_build_image_from_base
    )
