#!/usr/bin/env python3
"""
Test checkpoint loading with the new standardized format
"""
import sys
import os

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from bot import latest_checkpoint
except ImportError as e:
    print(f"❌ Failed to import bot components: {e}")
    sys.exit(1)

def main():
    print("Testing checkpoint loading...")
    print("=" * 50)
    
    checkpoint_path = latest_checkpoint()
    
    if checkpoint_path:
        print(f"✅ Found checkpoint: {checkpoint_path}")
        
        # Check if file exists
        if os.path.exists(checkpoint_path):
            file_size = os.path.getsize(checkpoint_path)
            mod_time = os.path.getmtime(checkpoint_path)
            from datetime import datetime
            
            print(f"   Size: {file_size:,} bytes")
            print(f"   Modified: {datetime.fromtimestamp(mod_time)}")
            
            # Try to load the model to verify it's valid
            try:
                import torch
                model = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                print(f"   ✅ Model loaded successfully")
                print(f"   Model keys: {list(model.keys())}")
                
                # Check for expected layers
                output_layers = [k for k in model.keys() if 'output' in k.lower() or k.endswith('.weight')]
                if output_layers:
                    print(f"   Output layers found: {len(output_layers)}")
                    for layer in output_layers[:3]:  # Show first 3
                        print(f"     {layer}: {model[layer].shape}")
                
            except Exception as e:
                print(f"   ❌ Error loading model: {e}")
        else:
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
    else:
        print("❌ No checkpoint found")
        
        # List available checkpoints for debugging
        print("\nAvailable checkpoint directories:")
        checkpoint_root = os.path.join(os.path.dirname(__file__), 'data', 'checkpoints')
        if os.path.exists(checkpoint_root):
            for item in os.listdir(checkpoint_root):
                item_path = os.path.join(checkpoint_root, item)
                if os.path.isdir(item_path):
                    print(f"  {item}/")

if __name__ == '__main__':
    main()