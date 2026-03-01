#!/usr/bin/env python3
"""
YouTube Shorts AI Personalizer - Startup Script
This script installs dependencies and starts the backend.
Designed to be double-click friendly on Windows/macOS/Linux.
"""

import os
import subprocess
import sys
import time

REQUIRED_PACKAGES = ["fastapi", "uvicorn", "pydantic"]


def run_command(command, description):
    """Run a command and stream output so users can see progress/errors."""
    print(f"\n🔄 {description}...")
    try:
        subprocess.run(command, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed (exit code {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found while {description}")
        return False


def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def install_dependencies():
    """Install required Python packages using the active interpreter."""
    for package in REQUIRED_PACKAGES:
        print(f"\n📦 Installing {package}...")
        # Prefer interpreter-tied pip so double-click runs install into the right env.
        if run_command([sys.executable, "-m", "pip", "install", package], f"Installing {package}"):
            continue

        # Fallbacks for systems where python -m pip is unavailable.
        if run_command(["pip", "install", package], f"Installing {package} with pip"):
            continue
        if run_command(["pip3", "install", package], f"Installing {package} with pip3"):
            continue

        print(f"❌ Failed to install {package}")
        return False

    return True


def start_application():
    print("\n🚀 Starting YouTube Shorts AI Personalizer backend...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔧 Keep this window open while using the extension")
    print("\n" + "=" * 60)

    # Prefer module invocation to avoid PATH issues with `uvicorn` command.
    if run_command([sys.executable, "-m", "uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
                   "Starting backend with uvicorn"):
        return True

    # Final fallback
    return run_command([sys.executable, "app.py"], "Starting backend via app.py")


def pause_before_exit():
    """Keep output visible when script is launched by double-click."""
    # Windows double-click launches often have no interactive stdin.
    if os.name == "nt":
        os.system("pause")
        return

    if sys.stdin and sys.stdin.isatty():
        try:
            input("\nPress Enter to close...")
            return
        except EOFError:
            pass

    print("\nClosing in 15 seconds...")
    time.sleep(15)


def main():
    print("🎬 YouTube Shorts AI Personalizer - Startup Script")
    print("=" * 60)

    if not check_python_version():
        return 1

    if not os.path.exists("app.py"):
        print("❌ app.py not found. Run this script from the project directory.")
        return 1

    if not os.path.exists("model.py"):
        print("❌ model.py not found. Run this script from the project directory.")
        return 1

    print("✅ Project files found")

    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        print("   Try manually: python -m pip install fastapi uvicorn pydantic")
        return 1

    print("\n✅ All dependencies installed successfully")
    time.sleep(1)

    if not start_application():
        print("\n❌ Failed to start backend.")
        return 1

    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌ Unexpected startup error: {exc}")
        code = 1
    finally:
        pause_before_exit()

    raise SystemExit(code)
