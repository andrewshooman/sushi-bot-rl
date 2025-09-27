#!/usr/bin/env python3
"""
Simple test script to verify our observation and action parsers work
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

try:
    from train import ExampleBotObs, ExampleBotAction, build_rlgym_v2_env
    print("✅ Successfully imported components")
    
    # Test creating the environment
    print("🔧 Creating environment...")
    env = build_rlgym_v2_env()
    print("✅ Environment created successfully")
    
    # Test reset
    print("🔄 Testing environment reset...")
    obs, info = env.reset()
    print(f"✅ Environment reset successful. Observation shape: {obs.shape if hasattr(obs, 'shape') else 'dict with keys: ' + str(list(obs.keys()))}")
    
    # Test action space
    print("🎮 Testing action space...")
    action_space = env.action_space
    print(f"✅ Action space: {action_space}")
    
    # Test a single step
    print("👟 Testing single step...")
    import numpy as np
    
    # The environment expects actions for multiple agents (blue + orange)
    # With ExampleBotAction, we have Discrete(8) action space
    single_action = 0  # Valid action from 0-7
    
    # Create action array for all agents (2 agents in this case)
    num_agents = len(obs) if isinstance(obs, dict) else 2
    print(f"Creating actions for {num_agents} agents")
    action = np.array([single_action] * num_agents, dtype=np.int32)
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Step successful. Reward: {reward}")
    
    print("\n🎉 All tests passed! The training setup should work now.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()