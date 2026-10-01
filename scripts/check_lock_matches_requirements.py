"""requirements.lock must satisfy requirements.txt.

CI, the security scans and the image all install from the hash-locked requirements.lock. A
change to requirements.txt alone (which is what Dependabot opens) therefore changes nothing that
is tested: on 2026-10-01 five such bumps were merged on green CI, and the versions they named
had never run. This check makes that state visible: if requirements.txt asks for something the
lock does not pin, the build fails until the lock is regenerated

    python -m piptools compile --generate-hashes --output-file=requirements.lock requirements.txt

and the suite has actually run on the new versions.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==([^\s\\]+)")


def locked(path: Path) -> dict[str, Version]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PIN.match(line)
        if match:
            pins[canonicalize_name(match.group(1))] = Version(match.group(2))
    return pins


def required(path: Path) -> list[Requirement]:
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            out.append(Requirement(line))
    return out


def problems(requirements: Path, lock: Path) -> list[str]:
    pins = locked(lock)
    found = []
    for requirement in required(requirements):
        name = canonicalize_name(requirement.name)
        if name not in pins:
            found.append(f"{requirement.name}: required, but not pinned in {lock.name}")
        elif not requirement.specifier.contains(pins[name], prereleases=True):
            found.append(
                f"{requirement.name}: {requirements.name} asks for {requirement.specifier}, "
                f"{lock.name} pins {pins[name]}"
            )
    return found


def main() -> int:
    found = problems(ROOT / "requirements.txt", ROOT / "requirements.lock")
    if found:
        print("requirements.lock does not satisfy requirements.txt.")
        print("Regenerate the lock and run the suite on it:")
        for line in found:
            print(f"  - {line}")
        return 1
    print("requirements.lock satisfies every requirement in requirements.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
