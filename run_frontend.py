"""
Frontend Launcher for ADR Prediction System
"""

import webbrowser
import os
from pathlib import Path

def launch_frontend():
    """Launch the frontend in browser"""
    # Get the frontend file path
    frontend_path = Path(__file__).parent / "frontend" / "standalone.html"
    
    if frontend_path.exists():
        # Convert to file URL
        file_url = f"file:///{frontend_path.as_posix()}"
        
        print("🌐 Launching ADR Prediction System Frontend...")
        print(f"📁 File: {frontend_path}")
        print(f"🔗 URL: {file_url}")
        
        # Open in browser
        webbrowser.open(file_url)
        
        print("✅ Frontend launched in default browser")
        print("🚀 The system is ready for use!")
        print("\n📋 Features Available:")
        print("  • Real-time ADR prediction")
        print("  • Clinical risk assessment")
        print("  • SHAP feature explanations")
        print("  • Interactive visualizations")
        print("  • Sample data loading")
        print("\n🔗 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        
    else:
        print(f"❌ Frontend file not found: {frontend_path}")
        print("Please ensure the file exists before running.")

if __name__ == "__main__":
    launch_frontend()
