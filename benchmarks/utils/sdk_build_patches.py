"""
Host-side patches for the bundled software-agent-sdk.

The SDK's ``_extract_tarball`` in ``openhands.agent_server.docker.build`` uses
``contextlib.chdir(dest)`` plus ``tar.extractall(path=".")``. Because ``chdir``
mutates the process-wide cwd, concurrent workers building docker contexts in
parallel race each other: one worker's extraction writes into another's tmp
dir, and cleanup of one tmp dir can leave another worker holding a dangling
cwd that breaks the next ``os.getcwd()`` (typically inside ``Path.absolute``
during pydantic validation).

The fix passes ``path=dest`` directly to ``extractall`` and removes the
``chdir`` wrapper, making ``_extract_tarball`` cwd-independent and safe to
run from multiple threads.
"""

from __future__ import annotations

import tarfile
from pathlib import Path


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


def apply_host_patches() -> None:
    from openhands.agent_server.docker import build

    build._extract_tarball = _safe_extract_tarball
