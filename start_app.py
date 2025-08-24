#!/usr/bin/env python3
"""
YouTube Shorts AI Personalizer - Startup Script
This script automatically installs dependencies and starts the application.
"""

import subprocess
import sys
import os
import time

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required Python packages."""
    packages = ["fastapi", "uvicorn"]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if not run_command(f"pip install {package}", f"Installing {package}"):
            # Try with pip3 if pip fails
            if not run_command(f"pip3 install {package}", f"Installing {package} with pip3"):
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def start_application():
    """Start the FastAPI application."""
    print("\n🚀 Starting YouTube Shorts AI Personalizer backend...")
    print("📍 Server will be available at: http://localhost:8000" )
    print("🔧 Press Ctrl+C to stop the server")
    print("\n" + "="*50)
    
    try:
        # Try uvicorn first
        subprocess.run(["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback to python app.py
            subprocess.run([sys.executable, "app.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start application: {e}")
            return False
    
    return True

def main():
    """Main startup function."""
    print("🎬 YouTube Shorts AI Personalizer - Startup Script")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("❌ app.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    if not os.path.exists("model.py"):
        print("❌ model.py not found. Please run this script from the project directory.")
        sys.exit(1)
    
    print("✅ Project files found")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please install manually:")
        print("   pip install fastapi uvicorn")
        sys.exit(1)
    
    print("\n✅ All dependencies installed successfully!")
    
    # Wait a moment
    time.sleep(1)
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()
