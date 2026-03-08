#!/usr/bin/env python3
"""
YouTube Shorts AI Personalizer - Startup Script.
Installs dependencies if needed and starts the backend.
Designed to be double-click friendly on Windows/macOS/Linux.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REQUIRED_PACKAGES = ["fastapi", "uvicorn", "pydantic"]
PROJECT_DIR = Path(__file__).resolve().parent


def run_command(command: Sequence[str], description: str, cwd: Path | None = None) -> bool:
    """Run a command and stream output so users can see progress/errors."""
    print(f"\n[run] {description}...")
    try:
        subprocess.run(command, check=True, cwd=str(cwd or PROJECT_DIR))
        print(f"[ok] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[error] {description} failed (exit code {exc.returncode})")
        return False
    except FileNotFoundError:
        print(f"[error] Command not found while {description}")
        return False


def check_python_version() -> bool:
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("[error] Python 3.8 or higher is required")
        return False
    print(f"[ok] Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies() -> bool:
    """Install required Python packages using the active interpreter."""
    for package in REQUIRED_PACKAGES:
        print(f"\n[pkg] Installing {package}...")
        if run_command([sys.executable, "-m", "pip", "install", package], f"Installing {package}"):
            continue
        if run_command(["pip", "install", package], f"Installing {package} with pip"):
            continue
        if run_command(["pip3", "install", package], f"Installing {package} with pip3"):
            continue
        print(f"[error] Failed to install {package}")
        return False
    return True


def start_application() -> bool:
    print("\n[start] Starting YouTube Shorts AI Personalizer backend...")
    print("[info] Server will be available at: http://localhost:8000")
    print("[info] Keep this window open while using the extension")
    print("\n" + "=" * 60)

    # Avoid --reload here because reloader subprocesses can exit immediately in double-click launches.
    if run_command(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        "Starting backend with uvicorn",
    ):
        return True

    return run_command([sys.executable, "app.py"], "Starting backend via app.py")


def pause_before_exit() -> None:
    """Keep output visible when the script is launched by double-click."""
    if os.name == "nt":
        try:
            subprocess.run(["cmd", "/k", "echo. & echo Press any key to close this window... & pause >nul"], check=False)
        except Exception:
            pass
        return

    if sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to close...")
            return
        except EOFError:
            pass

    print("\nClosing in 15 seconds...")
    time.sleep(15)


def main() -> int:
    os.chdir(PROJECT_DIR)
    print("YouTube Shorts AI Personalizer - Startup Script")
    print("=" * 60)
    print(f"[info] Working directory: {PROJECT_DIR}")

    if not check_python_version():
        return 1

    if not (PROJECT_DIR / "app.py").exists():
        print("[error] app.py not found next to start_app.py")
        return 1

    if not (PROJECT_DIR / "model.py").exists():
        print("[error] model.py not found next to start_app.py")
        return 1

    print("[ok] Project files found")

    print("\n[setup] Installing dependencies...")
    if not install_dependencies():
        print("\n[error] Failed to install dependencies.")
        print("Try manually: python -m pip install fastapi uvicorn pydantic")
        return 1

    print("\n[ok] All dependencies installed successfully")
    time.sleep(1)

    if not start_application():
        print("\n[error] Failed to start backend.")
        return 1

    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[error] Unexpected startup error: {exc}")
        code = 1
    finally:
        pause_before_exit()

    raise SystemExit(code)
