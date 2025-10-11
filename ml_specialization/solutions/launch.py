#!/usr/bin/env python3
"""
ML Specialization Solutions Launcher
Quick setup and launch script for all solutions
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print welcome banner"""
    print("🚀" + "="*60 + "🚀")
    print("   ML SPECIALIZATION SOLUTIONS LAUNCHER")
    print("   25 Complete Jupyter Notebook Solutions")
    print("🚀" + "="*60 + "🚀")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    if python_version < (3, 7):
        print("❌ Error: Python 3.7 or higher required")
        print(f"   Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    return True

def check_pip():
    """Check if pip is available"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error: pip not found")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        
        print("✅ All packages installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def launch_jupyter():
    """Launch Jupyter Lab"""
    print("\n🚀 Launching Jupyter Lab...")
    print("📝 Your solutions will open in your default browser")
    print("🔗 If browser doesn't open automatically, copy the URL from the terminal")
    print("\n" + "="*50)
    
    try:
        # Launch Jupyter Lab
        subprocess.run([sys.executable, "-m", "jupyter", "lab"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Error launching Jupyter Lab")
        print("💡 Try: pip install jupyterlab")
        return False
    except KeyboardInterrupt:
        print("\n👋 Jupyter Lab stopped by user")
        return True

def show_quick_start():
    """Show quick start guide"""
    print("\n📚 QUICK START GUIDE")
    print("="*40)
    print("1. 🏠 Start with Assignment 1 (House Price Prediction)")
    print("2. 🏦 Continue with Assignment 2 (Banking Analytics)")  
    print("3. 🧠 Try Assignment 3 (Neural Networks)")
    print("4. 📈 Progress through all 25 assignments")
    print()
    print("💡 TIP: Each notebook is self-contained and ready to run!")
    print("🎯 GOAL: Complete all 25 assignments for ML mastery")
    print()

def show_available_solutions():
    """Show list of available solutions"""
    print("📋 AVAILABLE SOLUTIONS:")
    print("-" * 50)
    
    solutions = []
    for i in range(1, 26):
        filename = f"assignment_{i:02d}_solution.ipynb"
        if os.path.exists(filename):
            solutions.append(f"✅ Assignment {i:2d}: {filename}")
        else:
            solutions.append(f"❌ Assignment {i:2d}: {filename} (missing)")
    
    for solution in solutions:
        print(solution)
    
    found_count = len([s for s in solutions if "✅" in s])
    print(f"\n📊 Found {found_count}/25 solution files")
    
    if found_count < 25:
        print("🔧 Run generate_solutions.py to create missing solutions")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_pip():
        sys.exit(1)
    
    # Show available solutions
    show_available_solutions()
    
    # Ask user what they want to do
    print("\n🎯 What would you like to do?")
    print("1. 📦 Install requirements and launch Jupyter Lab")
    print("2. 🚀 Just launch Jupyter Lab (requirements already installed)")
    print("3. 📋 Show setup instructions")
    print("4. 🔧 Regenerate all solutions")
    print("5. ❌ Exit")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            if install_requirements():
                show_quick_start()
                launch_jupyter()
            break
            
        elif choice == "2":
            show_quick_start()
            launch_jupyter()
            break
            
        elif choice == "3":
            print("\n📖 SETUP INSTRUCTIONS:")
            print("="*40)
            print("1. Make sure you have Python 3.7+ installed")
            print("2. Run: pip install -r requirements.txt")
            print("3. Run: jupyter lab")
            print("4. Open any assignment_XX_solution.ipynb file")
            print("5. Follow the notebook instructions")
            print("\n💡 For detailed setup, see SETUP_GUIDE.md")
            break
            
        elif choice == "4":
            print("\n🔧 Regenerating all solutions...")
            try:
                subprocess.run([sys.executable, "generate_solutions.py"], check=True)
                print("✅ All solutions regenerated!")
            except subprocess.CalledProcessError:
                print("❌ Error regenerating solutions")
            break
            
        elif choice == "5":
            print("\n👋 Thanks for using ML Specialization Solutions!")
            print("🎓 Happy learning!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()