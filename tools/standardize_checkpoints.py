#!/usr/bin/env python3
"""
Checkpoint Standardization Script
Reorganizes all checkpoints into a consistent format for easy loading by bot.py
"""

import os
import shutil
import glob
from datetime import datetime
import json

# Configuration
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(BOT_DIR, 'data', 'checkpoints')
STANDARDIZED_DIR = os.path.join(CHECKPOINT_ROOT, 'standardized')

def get_checkpoint_info(checkpoint_path):
    """Get information about a checkpoint"""
    try:
        # Get file modification time
        mod_time = os.path.getmtime(checkpoint_path)
        
        # Extract training steps from path
        path_parts = checkpoint_path.replace('\\', '/').split('/')
        steps = None
        for part in path_parts:
            if part.isdigit():
                steps = int(part)
                break
        
        # Extract model type from path
        model_type = 'unknown'
        if 'aggressive' in checkpoint_path.lower():
            model_type = 'aggressive'
        elif 'enhanced' in checkpoint_path.lower():
            model_type = 'enhanced'
        elif 'simple' in checkpoint_path.lower():
            model_type = 'simple'
        
        return {
            'path': checkpoint_path,
            'timestamp': mod_time,
            'datetime': datetime.fromtimestamp(mod_time),
            'steps': steps,
            'model_type': model_type
        }
    except Exception as e:
        print(f"Error processing {checkpoint_path}: {e}")
        return None

def find_all_checkpoints():
    """Find all PPO_POLICY.pt files in the checkpoint directory"""
    pattern = os.path.join(CHECKPOINT_ROOT, '**', 'PPO_POLICY.pt')
    checkpoint_files = glob.glob(pattern, recursive=True)
    
    # Get info for each checkpoint
    checkpoints = []
    for cp_file in checkpoint_files:
        info = get_checkpoint_info(cp_file)
        if info:
            checkpoints.append(info)
    
    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
    return checkpoints

def standardize_checkpoints():
    """Reorganize checkpoints into standardized format"""
    print("Finding all checkpoints...")
    checkpoints = find_all_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # Create standardized directory
    os.makedirs(STANDARDIZED_DIR, exist_ok=True)
    
    # Create subdirectories for each model type
    for model_type in ['aggressive', 'enhanced', 'simple', 'unknown']:
        os.makedirs(os.path.join(STANDARDIZED_DIR, model_type), exist_ok=True)
    
    print(f"Found {len(checkpoints)} checkpoints")
    print("\\nProcessing checkpoints...")
    
    # Process each checkpoint
    for i, cp_info in enumerate(checkpoints):
        model_type = cp_info['model_type']
        steps = cp_info['steps'] or 0
        timestamp = cp_info['datetime'].strftime('%Y%m%d_%H%M%S')
        
        # Create standardized name: YYYYMMDD_HHMMSS_steps_rank
        rank = i + 1  # 1 = newest, 2 = second newest, etc.
        std_name = f"{timestamp}_{steps:08d}_{rank:03d}"
        
        # Create destination directory
        dest_dir = os.path.join(STANDARDIZED_DIR, model_type, std_name)
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy all files from source directory
        source_dir = os.path.dirname(cp_info['path'])
        try:
            for file_name in os.listdir(source_dir):
                source_file = os.path.join(source_dir, file_name)
                dest_file = os.path.join(dest_dir, file_name)
                
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, dest_file)
            
            print(f"  {rank:3d}. {model_type:10s} | {steps:8d} steps | {timestamp} -> {std_name}")
            
        except Exception as e:
            print(f"  ERROR copying {source_dir}: {e}")
    
    # Create latest symlinks
    create_latest_symlinks()
    
    # Generate summary
    generate_summary(checkpoints)

def create_latest_symlinks():
    """Create 'latest' directories pointing to the newest checkpoint of each type"""
    print("\\nCreating latest symlinks...")
    
    for model_type in ['aggressive', 'enhanced', 'simple']:
        type_dir = os.path.join(STANDARDIZED_DIR, model_type)
        if not os.path.exists(type_dir):
            continue
            
        # Find newest checkpoint for this type
        subdirs = [d for d in os.listdir(type_dir) if os.path.isdir(os.path.join(type_dir, d))]
        if not subdirs:
            continue
            
        # Sort by rank (lowest number = newest) - rank is at the end after last underscore
        subdirs.sort(key=lambda x: int(x.split('_')[-1]))
        newest = subdirs[0]
        
        # Create latest directory
        latest_dir = os.path.join(STANDARDIZED_DIR, f"latest_{model_type}")
        if os.path.exists(latest_dir):
            shutil.rmtree(latest_dir)
        
        # Copy instead of symlink (Windows compatibility)
        source_path = os.path.join(type_dir, newest)
        shutil.copytree(source_path, latest_dir)
        
        print(f"  latest_{model_type} -> {newest}")

def generate_summary(checkpoints):
    """Generate a summary JSON file"""
    summary = {
        'generated': datetime.now().isoformat(),
        'total_checkpoints': len(checkpoints),
        'checkpoints': []
    }
    
    for i, cp_info in enumerate(checkpoints):
        rank = i + 1
        summary['checkpoints'].append({
            'rank': rank,
            'model_type': cp_info['model_type'],
            'steps': cp_info['steps'],
            'timestamp': cp_info['datetime'].isoformat(),
            'original_path': cp_info['path']
        })
    
    # Save summary
    summary_path = os.path.join(STANDARDIZED_DIR, 'checkpoint_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\nSummary saved to: {summary_path}")

def main():
    print("Rocket League Bot Checkpoint Standardization")
    print("=" * 50)
    print(f"Checkpoint root: {CHECKPOINT_ROOT}")
    print(f"Standardized dir: {STANDARDIZED_DIR}")
    print()
    
    # Ask for confirmation
    response = input("This will reorganize all checkpoints. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    standardize_checkpoints()
    
    print("\\n" + "=" * 50)
    print("Standardization complete!")
    print()
    print("Latest checkpoints are available at:")
    for model_type in ['aggressive', 'enhanced', 'simple']:
        latest_path = os.path.join(STANDARDIZED_DIR, f"latest_{model_type}")
        if os.path.exists(latest_path):
            print(f"  {latest_path}")
    print()
    print("Next step: Update bot.py to use standardized checkpoints")

if __name__ == '__main__':
    main()