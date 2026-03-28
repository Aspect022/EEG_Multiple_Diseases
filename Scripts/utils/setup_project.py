import os
import subprocess
import sys

def run_command(command, description):
    print(f"\n--- {description} ---")
    try:
        subprocess.check_call(command, shell=True)
        print("✅ SUCCESS")
    except subprocess.CalledProcessError as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

def main():
    print("🚀 INITIALIZING RTX 5050 ECG PIPELINE ENVIRONMENT 🚀")
    
    # Check if inside venv
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print("⚠️ WARNING: You are not running inside a virtual environment.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Setup aborted.")
            sys.exit(1)
            
    # Install PyTorch
    print("\nLooking for compatible PyTorch (assuming CUDA 12.1 for modern RTX cards)...")
    run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Installing PyTorch Platform"
    )
    
    # Install main requirements
    run_command("pip install -r requirements.txt", "Installing General Dependencies")
    
    print("\n🎉 ENVIRONMENT SETUP COMPLETE! 🎉")
    print("You can now run the pipeline using:")
    print("python unified_pipeline.py\n")

if __name__ == "__main__":
    main()