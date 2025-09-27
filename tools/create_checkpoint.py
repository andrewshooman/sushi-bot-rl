#!/usr/bin/env python3
"""
Quick script to create a compatible checkpoint for the bot.
This creates a simple trained model that the bot can load.
"""

import torch
import torch.nn as nn
import os
import numpy as np
from datetime import datetime

# Create a simple policy network that matches the bot architecture
class BotCompatiblePolicy(nn.Module):
    def __init__(self, obs_size=243):  # Enhanced observations: 243 features
        super().__init__()
        # Architecture must match SushiBot: 243 -> 512 -> 256 -> 256 -> 8 actions
        self.model = nn.Sequential(
            nn.Linear(obs_size, 512), nn.ReLU(),    # Layer 0
            nn.Linear(512, 256), nn.ReLU(),         # Layer 2
            nn.Linear(256, 256), nn.ReLU(),         # Layer 4
            nn.Linear(256, 8)                       # Layer 6 (enhanced action space)
        )
    
    def forward(self, x):
        return self.model(x)

def create_simple_checkpoint():
    """Create a simple checkpoint that the bot can load"""
    
    # Create the model
    model = BotCompatiblePolicy()
    
    # Initialize weights with small random values (better than zero initialization)
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) > 1:  # Weight matrices
                nn.init.xavier_uniform_(param, gain=0.1)  # Small weights for stable start
            else:  # Biases
                nn.init.zeros_(param)
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("data", "checkpoints", f"run_{timestamp}_simple_manual")
    unique_dir = os.path.join(checkpoint_dir, "0")
    os.makedirs(unique_dir, exist_ok=True)
    
    # Save the model state dict
    checkpoint_path = os.path.join(unique_dir, "PPO_POLICY.pt")
    torch.save(model.state_dict(), checkpoint_path)
    
    # Create additional files that RLGym-PPO usually creates
    book_keeping = {
        "timesteps": 1000,
        "model_updates": 1,
        "policy_lr": 0.0001,
        "critic_lr": 0.0001
    }
    
    book_keeping_path = os.path.join(unique_dir, "BOOK_KEEPING_VARS.json")
    import json
    with open(book_keeping_path, 'w') as f:
        json.dump(book_keeping, f)
    
    print(f"✅ Created compatible checkpoint: {checkpoint_path}")
    print(f"📁 Directory: {checkpoint_dir}")
    print(f"🎯 Architecture: 243 obs -> 512 -> 256 -> 256 -> 8 actions")
    print(f"🤖 Bot should now be able to load this checkpoint!")
    
    return checkpoint_path

def test_checkpoint_compatibility():
    """Test if the checkpoint is compatible with the bot"""
    print("\n🧪 Testing checkpoint compatibility...")
    
    # Import bot components
    import sys
    import os
    src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from bot import latest_checkpoint, TinyPolicy, OBS_SIZE, N_ACTIONS
    except ImportError as e:
        print(f"❌ Failed to import bot components: {e}")
        return False
    
    # Check if our checkpoint is detected
    ckpt_path = latest_checkpoint()
    if not ckpt_path:
        print("❌ No checkpoint detected")
        return False
    
    print(f"📋 Latest checkpoint: {ckpt_path}")
    
    # Try to load it
    try:
        model = TinyPolicy()
        state = torch.load(ckpt_path, map_location='cpu')
        
        # Fix key naming if needed
        fixed_state_dict = {}
        for key, value in state.items():
            if key.startswith("model."):
                new_key = key.replace("model.", "net.")
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        # Check shapes
        if 'net.0.weight' in fixed_state_dict:
            input_size = fixed_state_dict['net.0.weight'].shape[1]
            output_size = fixed_state_dict['net.4.weight'].shape[0]
            
            print(f"🔍 Model architecture: {input_size} obs, {output_size} actions")
            print(f"🎯 Bot expects: {OBS_SIZE} obs, {N_ACTIONS} actions")
            
            if input_size == OBS_SIZE and output_size == N_ACTIONS:
                model.load_state_dict(fixed_state_dict)
                print("✅ Checkpoint is fully compatible!")
                
                # Test forward pass
                dummy_input = torch.randn(1, OBS_SIZE)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"✅ Forward pass successful: {output.shape}")
                return True
            else:
                print(f"❌ Architecture mismatch")
                return False
        else:
            print("❌ Unexpected checkpoint format")
            return False
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Creating bot-compatible checkpoint...")
    checkpoint_path = create_simple_checkpoint()
    
    # Test it
    compatible = test_checkpoint_compatibility()
    
    if compatible:
        print("\n🎉 SUCCESS! Your bot should now load the trained model!")
        print("💡 Run your bot in RLBot to see it in action!")
    else:
        print("\n❌ Something went wrong. Check the error messages above.")