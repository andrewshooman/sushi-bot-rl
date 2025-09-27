"""
Enhanced action parser based on Nexto's sophisticated lookup table approach
Handles ground and aerial actions separately with proper boost/jump logic
"""
import numpy as np
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import math


class EnhancedActionParser:
    """
    Nexto-style action parser using lookup table for discrete actions
    Separates ground and aerial actions with proper boost/jump logic
    """
    
    def __init__(self):
        self._lookup_table = self.make_lookup_table()
        self.num_actions = len(self._lookup_table)
        
    @staticmethod
    def make_lookup_table():
        """Create lookup table for all possible actions (based on Nexto)"""
        actions = []
        
        # Ground actions
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue  # Can only boost when throttling forward
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        
        # Aerial actions
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        
        actions = np.array(actions, dtype=np.float32)
        return actions
    
    def get_action_space_size(self):
        """Get the number of discrete actions"""
        return self.num_actions
    
    def parse_action_index(self, action_index: int) -> np.ndarray:
        """Convert action index to continuous control values"""
        if 0 <= action_index < len(self._lookup_table):
            return self._lookup_table[action_index].copy()
        else:
            # Default action (idle)
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def parse_action_logits(self, logits: torch.Tensor, beta: float = 1.0) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Parse action logits using Nexto's beta sampling approach
        
        Args:
            logits: Action logits from neural network
            beta: Sampling parameter (1=deterministic, 0.5=balanced, 0=random, -1=worst)
            
        Returns:
            Tuple of (parsed_action, weights/attention if available)
        """
        
        # Handle different output formats
        if isinstance(logits, tuple):
            out, weights = logits
            out = (out,)
        else:
            out = (logits,)
            weights = None
        
        # Pad logits to consistent size
        max_shape = max(o.shape[-1] for o in out)
        padded_logits = torch.stack([
            l if l.shape[-1] == max_shape
            else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
            for l in out
        ], dim=1)
        
        # Apply beta sampling
        if beta == 1:
            # Deterministic - best action
            action_indices = torch.argmax(padded_logits, dim=-1)
        elif beta == -1:
            # Anti-deterministic - worst action
            action_indices = torch.argmin(padded_logits, dim=-1)
        else:
            # Stochastic sampling
            if beta == 0:
                # Pure random
                padded_logits[torch.isfinite(padded_logits)] = 0
            else:
                # Scaled sampling
                padded_logits *= math.log((beta + 1) / (1 - beta), 3)
            
            dist = Categorical(logits=padded_logits)
            action_indices = dist.sample()
        
        # Convert to numpy and parse
        action_index = action_indices.numpy().item() if hasattr(action_indices, 'numpy') else action_indices
        parsed_action = self.parse_action_index(action_index)
        
        return parsed_action, weights


class SimpleActionParser:
    """
    Simplified action parser for backward compatibility
    Maps neural network outputs to basic discrete actions
    """
    
    def __init__(self):
        # Simple 9-action space: combinations of throttle, steer, jump, boost
        self.actions = np.array([
            # [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
            [0, 0, 0, 0, 0, 0, 0, 0],    # 0: No action
            [1, 0, 0, 0, 0, 0, 0, 0],    # 1: Forward
            [-1, 0, 0, 0, 0, 0, 0, 0],   # 2: Backward
            [1, -1, 0, 0, 0, 0, 0, 0],   # 3: Forward + Left
            [1, 1, 0, 0, 0, 0, 0, 0],    # 4: Forward + Right
            [1, 0, 0, 0, 0, 1, 0, 0],    # 5: Forward + Jump
            [1, 0, 0, 0, 0, 0, 1, 0],    # 6: Forward + Boost
            [1, -1, 0, 0, 0, 0, 1, 0],   # 7: Forward + Left + Boost  
            [1, 1, 0, 0, 0, 0, 1, 0],    # 8: Forward + Right + Boost
        ], dtype=np.float32)
    
    def get_action_space_size(self):
        """Get the number of discrete actions"""
        return len(self.actions)
    
    def parse_action_index(self, action_index: int) -> np.ndarray:
        """Convert action index to continuous control values"""
        if 0 <= action_index < len(self.actions):
            return self.actions[action_index].copy()
        else:
            return self.actions[0].copy()  # Default to no action


def create_action_parser(enhanced: bool = True) -> "ActionParser":
    """Factory function to create appropriate action parser"""
    if enhanced:
        return EnhancedActionParser()
    else:
        return SimpleActionParser()