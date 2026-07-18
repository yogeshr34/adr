"""
Complete End-to-End ADR Prediction System Launcher
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    print("🔍 Checking system requirements...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'scikit-learn', 
        'numpy', 'pandas', 'shap', 'matplotlib', 'seaborn'
    ]
    
    optional_packages = ['qiskit', 'qiskit-machine-learning', 'qiskit-algorithms']
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"❌ {package}")
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} (optional)")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠️ {package} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {', '.join(missing_required)}")
        print("Install with: pip install -r requirements_complete.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
        print("Quantum features will be limited without these packages.")
    
    print("\n✅ System requirements check complete!")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_complete.txt"
        ])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    
    # Change to backend directory
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "complete_app.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped by user")
        return True

def open_frontend():
    """Open the frontend in browser"""
    print("🌐 Opening frontend in browser...")
    
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    
    if frontend_path.exists():
        # Open the HTML file in browser
        webbrowser.open(f'file://{frontend_path.absolute()}')
        print(f"✅ Frontend opened: {frontend_path.absolute()}")
        return True
    else:
        print(f"❌ Frontend file not found: {frontend_path}")
        return False

def main():
    """Main launcher function"""
    print("🎯 Complete End-to-End ADR Prediction System")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n📦 Installing missing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually.")
            return
    
    print("\n🌟 Starting complete system...")
    
    # Open frontend in a separate thread
    frontend_thread = threading.Thread(target=open_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()
    
    # Wait a moment for browser to open
    time.sleep(2)
    
    # Start backend (this will block)
    print("\n🔧 Backend server starting on http://localhost:8000")
    print("📱 Frontend should open in your browser")
    print("🛑 Press Ctrl+C to stop the system")
    
    try:
        start_backend()
    except KeyboardInterrupt:
        print("\n👋 System shutdown complete!")
    except Exception as e:
        print(f"\n❌ System error: {e}")

if __name__ == "__main__":
    main()
