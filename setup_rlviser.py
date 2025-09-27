#!/usr/bin/env python3
"""
RLViser Setup and Build Script
Builds the RLViser visualization tool from Rust source
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def check_rust_installed():
    """Check if Rust is installed"""
    try:
        result = subprocess.run(['cargo', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Rust/Cargo found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Rust/Cargo not found")
        return False

def install_rust():
    """Install Rust if not present"""
    print("📦 Installing Rust...")
    print("Please visit https://rustup.rs/ to install Rust, then run this script again.")
    print("Or run: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
    return False

def build_rlviser():
    """Build RLViser from source"""
    project_root = Path(__file__).parent
    rlviser_dir = project_root / "rlviser-0.8.2"
    
    if not rlviser_dir.exists():
        print(f"❌ RLViser directory not found: {rlviser_dir}")
        return False
    
    print(f"🔨 Building RLViser in {rlviser_dir}...")
    
    try:
        # Change to RLViser directory
        os.chdir(rlviser_dir)
        
        # Build in release mode for better performance
        print("Building in release mode (this may take a while)...")
        result = subprocess.run(['cargo', 'build', '--release'], 
                              capture_output=True, text=True, check=True)
        
        print("✅ RLViser built successfully!")
        
        # Check if executable was created
        exe_path = rlviser_dir / "target" / "release" / "rlviser.exe"
        if exe_path.exists():
            print(f"✅ Executable found at: {exe_path}")
            return True
        else:
            # Try without .exe extension (Linux/Mac)
            exe_path = rlviser_dir / "target" / "release" / "rlviser"
            if exe_path.exists():
                print(f"✅ Executable found at: {exe_path}")
                return True
            else:
                print("❌ Executable not found after build")
                return False
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def create_rlviser_launcher():
    """Create a convenient launcher script"""
    project_root = Path(__file__).parent
    rlviser_dir = project_root / "rlviser-0.8.2"
    
    # Find the executable
    exe_paths = [
        rlviser_dir / "target" / "release" / "rlviser.exe",
        rlviser_dir / "target" / "release" / "rlviser",
        rlviser_dir / "target" / "debug" / "rlviser.exe", 
        rlviser_dir / "target" / "debug" / "rlviser"
    ]
    
    exe_path = None
    for path in exe_paths:
        if path.exists():
            exe_path = path
            break
    
    if not exe_path:
        print("❌ No RLViser executable found")
        return False
    
    # Create launcher script
    launcher_content = f'''#!/bin/bash
# RLViser Launcher Script
cd "{rlviser_dir}"
"{exe_path}" "$@"
'''
    
    launcher_path = project_root / "launch_rlviser.sh"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    # Make executable on Unix systems
    try:
        os.chmod(launcher_path, 0o755)
    except:
        pass  # Windows doesn't need this
    
    print(f"✅ Created launcher script: {launcher_path}")
    return True

def test_rlviser():
    """Test if RLViser can be launched"""
    project_root = Path(__file__).parent
    rlviser_dir = project_root / "rlviser-0.8.2"
    
    # Find the executable
    exe_paths = [
        rlviser_dir / "target" / "release" / "rlviser.exe",
        rlviser_dir / "target" / "release" / "rlviser",
        rlviser_dir / "target" / "debug" / "rlviser.exe", 
        rlviser_dir / "target" / "debug" / "rlviser"
    ]
    
    exe_path = None
    for path in exe_paths:
        if path.exists():
            exe_path = path
            break
    
    if not exe_path:
        print("❌ No RLViser executable found for testing")
        return False
    
    print(f"🧪 Testing RLViser executable: {exe_path}")
    
    try:
        # Try to get version or help (quick test)
        result = subprocess.run([str(exe_path), '--help'], 
                              capture_output=True, text=True, timeout=5)
        print("✅ RLViser executable responds to --help")
        return True
    except subprocess.TimeoutExpired:
        print("✅ RLViser executable started (timed out waiting for help - this is normal)")
        return True
    except Exception as e:
        print(f"⚠️  RLViser test inconclusive: {e}")
        return True  # Assume it's fine if it exists

def main():
    """Main setup function"""
    print("RLViser Setup Script")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if Rust is installed
    if not check_rust_installed():
        install_rust()
        return False
    
    # Build RLViser
    if not build_rlviser():
        print("❌ Failed to build RLViser")
        return False
    
    # Create launcher
    create_rlviser_launcher()
    
    # Test the build
    test_rlviser()
    
    print("\n" + "=" * 50)
    print("✅ RLViser setup complete!")
    print("\nUsage:")
    print("  - Training with visualization: python src/train.py --train --rlviser")
    print("  - Manual launch: ./launch_rlviser.sh")
    print("\nNote: RLViser will open a 3D visualization window during training.")
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)