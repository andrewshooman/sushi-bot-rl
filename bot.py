import os
import glob
import numpy as np
import torch

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

# ----------------- shared constants -----------------
OBS_SIZE = 64                 # match the enhanced preset (64 observations) 
N_ACTIONS = 19                # match the enhanced discrete action space (19 actions)

# Get absolute path to checkpoints directory
BOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_ROOT = os.path.join(BOT_DIR, 'data', 'checkpoints')
# ----------------------------------------------------

def latest_checkpoint():
    # Priority order: current_aggressive > current_enhanced > current_simple > timestamped directories
    priority_dirs = ['current_aggressive', 'current_enhanced', 'current_simple']
    
    # First check priority directories
    for dir_name in priority_dirs:
        run_dir = os.path.join(CHECKPOINT_ROOT, dir_name)
        if os.path.isdir(run_dir):
            unique_dirs = sorted(glob.glob(os.path.join(run_dir, '*')))
            for unique_dir in reversed(unique_dirs):
                if os.path.isdir(unique_dir):
                    policy_path = os.path.join(unique_dir, 'PPO_POLICY.pt')
                    if os.path.isfile(policy_path):
                        return policy_path
    
    # Fallback to all directories (for timestamped backups)
    runs = sorted(glob.glob(os.path.join(CHECKPOINT_ROOT, '*')))
    if not runs:
        return None
    
    # Check for RLGym-PPO format first (newer format)
    for run_dir in reversed(runs):  # Check newest first
        # Look for RLGym-PPO checkpoint structure: run_dir/unique_id/PPO_POLICY.pt
        unique_dirs = sorted(glob.glob(os.path.join(run_dir, '*')))
        for unique_dir in reversed(unique_dirs):
            if os.path.isdir(unique_dir):
                policy_path = os.path.join(unique_dir, 'PPO_POLICY.pt')
                if os.path.isfile(policy_path):
                    return policy_path
    
    # Fallback to original format
    for run_dir in reversed(runs):
        # prefer explicit latest.pt
        lp = os.path.join(run_dir, 'latest.pt')
        if os.path.isfile(lp):
            return lp
        # fallback to lexicographically-last *.pt
        cands = sorted(glob.glob(os.path.join(run_dir, '*.pt')))
        if cands:
            return cands[-1]
    
    return None

class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Match RLGym-PPO architecture: 64 → 512 → 256 → 256 → 9
        self.net = torch.nn.Sequential(
            torch.nn.Linear(OBS_SIZE, 512), torch.nn.ReLU(),    # Layer 0
            torch.nn.Linear(512, 256), torch.nn.ReLU(),         # Layer 2  
            torch.nn.Linear(256, 256), torch.nn.ReLU(),         # Layer 4
            torch.nn.Linear(256, N_ACTIONS)                     # Layer 6
        )
    def forward(self, x): return self.net(x)

def packet_to_obs(packet: GameTickPacket, index: int) -> np.ndarray:
    me = packet.game_cars[index]
    ball = packet.game_ball.physics
    car = me.physics

    # Enhanced feature vector with improved game awareness (64 observations)
    obs = []
    
    # === SELF (CAR) INFORMATION === (12 features)
    # Car location (3 features)
    obs.extend([car.location.x/4096, car.location.y/5120, car.location.z/2044])
    
    # Car velocity (3 features)
    obs.extend([car.velocity.x/2300, car.velocity.y/2300, car.velocity.z/2300])
    
    # Car rotation (3 features)
    obs.extend([car.rotation.pitch/3.14159, car.rotation.yaw/3.14159, car.rotation.roll/3.14159])
    
    # Car angular velocity (3 features) - Enhanced
    obs.extend([car.angular_velocity.x/3.14159, car.angular_velocity.y/3.14159, car.angular_velocity.z/3.14159])
    
    # === BALL INFORMATION === (9 features)
    # Ball location (3 features)
    obs.extend([ball.location.x/4096, ball.location.y/5120, ball.location.z/2044])
    
    # Ball velocity (3 features)
    obs.extend([ball.velocity.x/2300, ball.velocity.y/2300, ball.velocity.z/2300])
    
    # Ball angular velocity (3 features) - Enhanced
    obs.extend([ball.angular_velocity.x/3.14159, ball.angular_velocity.y/3.14159, ball.angular_velocity.z/3.14159])
    
    # === RELATIVE INFORMATION === (6 features)
    # Distance to ball
    ball_distance = ((ball.location.x - car.location.x)**2 + 
                    (ball.location.y - car.location.y)**2 + 
                    (ball.location.z - car.location.z)**2)**0.5
    obs.append(ball_distance / 4096)
    
    # Direction to ball (normalized) (3 features)
    if ball_distance > 0:
        obs.extend([(ball.location.x - car.location.x) / ball_distance,
                   (ball.location.y - car.location.y) / ball_distance,
                   (ball.location.z - car.location.z) / ball_distance])
    else:
        obs.extend([0.0, 0.0, 0.0])
    
    # Relative velocity (2 features)
    rel_vel_x = ball.velocity.x - car.velocity.x
    rel_vel_y = ball.velocity.y - car.velocity.y
    rel_vel_z = ball.velocity.z - car.velocity.z
    rel_speed = (rel_vel_x**2 + rel_vel_y**2 + rel_vel_z**2)**0.5
    obs.append(rel_speed / 2300)  # Total relative speed
    
    # Speed toward ball
    if ball_distance > 0:
        speed_toward = ((rel_vel_x * (ball.location.x - car.location.x) + 
                        rel_vel_y * (ball.location.y - car.location.y) + 
                        rel_vel_z * (ball.location.z - car.location.z)) / ball_distance) / 2300
    else:
        speed_toward = 0.0
    obs.append(speed_toward)
    
    # === CAR STATE === (6 features)
    # Basic states
    obs.append(1.0 if me.has_wheel_contact else 0.0)
    obs.append(me.boost / 100.0)
    obs.append(1.0 if me.jumped else 0.0)
    
    # Enhanced states
    obs.append(1.0 if me.is_super_sonic else 0.0)
    obs.append(1.0 if not me.has_wheel_contact else 0.0)  # Airborne
    obs.append(1.0 if not me.jumped and not me.has_wheel_contact else 0.0)  # Used flip
    
    # === BOOST AND POSITIONING === (6 features)
    # Distance to goal (approximate)
    goal_distance = (5120 - abs(car.location.y)) / 5120
    obs.append(goal_distance)
    
    # Ball height relative to car
    ball_height_rel = (ball.location.z - car.location.z) / 2044
    obs.append(ball_height_rel)
    
    # Car orientation relative to ball
    import math
    car_forward_x = math.cos(car.rotation.pitch) * math.cos(car.rotation.yaw)
    car_forward_y = math.cos(car.rotation.pitch) * math.sin(car.rotation.yaw)
    car_forward_z = math.sin(car.rotation.pitch)
    
    if ball_distance > 0:
        facing_ball = ((car_forward_x * (ball.location.x - car.location.x) + 
                       car_forward_y * (ball.location.y - car.location.y) + 
                       car_forward_z * (ball.location.z - car.location.z)) / ball_distance)
    else:
        facing_ball = 0.0
    obs.append(facing_ball)
    
    # Boost strategic information
    obs.append(me.boost / 33.0)  # Boost as fraction of supersonic threshold
    obs.append(1.0 if me.boost > 50 else 0.0)  # High boost indicator
    obs.append(car.location.x / 4096)  # Side position for boost pad awareness
    
    # === TEAMMATE/OPPONENT AWARENESS === (25 features)
    # Add basic awareness of other cars (simplified for 64-feature limit)
    for i in range(min(5, len(packet.game_cars))):  # Up to 5 other cars
        if i != index and i < len(packet.game_cars):
            other_car = packet.game_cars[i].physics
            # Distance to other car
            other_distance = ((other_car.location.x - car.location.x)**2 + 
                            (other_car.location.y - car.location.y)**2 + 
                            (other_car.location.z - car.location.z)**2)**0.5
            obs.append(other_distance / 4096)
            
            # Relative position (2 features)
            if other_distance > 0:
                obs.extend([(other_car.location.x - car.location.x) / other_distance,
                           (other_car.location.y - car.location.y) / other_distance])
            else:
                obs.extend([0.0, 0.0])
            
            # Other car velocity magnitude
            other_speed = (other_car.velocity.x**2 + other_car.velocity.y**2 + other_car.velocity.z**2)**0.5
            obs.append(other_speed / 2300)
            
            # Team indicator (1 if teammate, 0 if opponent)
            obs.append(1.0 if packet.game_cars[i].team == me.team else 0.0)
        else:
            # Pad with zeros if car doesn't exist
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Convert to numpy array and ensure correct size
    s = np.array(obs, dtype=np.float32)
    
    # Pad to OBS_SIZE or truncate if needed
    if s.shape[0] < OBS_SIZE:
        s = np.pad(s, (0, OBS_SIZE - s.shape[0]))
    return s[:OBS_SIZE]

# Enhanced action LUT with more turning options: (throttle, steer, boost, handbrake, jump)
LUT = [
    ( 1.0,  0.0, 0, 0, 0),  # 0: forward
    ( 1.0,  0.3, 0, 0, 0),  # 1: forward slight right
    ( 1.0,  0.6, 0, 0, 0),  # 2: forward medium right
    ( 1.0,  1.0, 0, 0, 0),  # 3: forward hard right
    ( 1.0, -0.3, 0, 0, 0),  # 4: forward slight left
    ( 1.0, -0.6, 0, 0, 0),  # 5: forward medium left  
    ( 1.0, -1.0, 0, 0, 0),  # 6: forward hard left
    ( 1.0,  0.0, 1, 0, 0),  # 7: forward + boost
    ( 1.0,  0.6, 1, 0, 0),  # 8: forward right + boost
    ( 1.0, -0.6, 1, 0, 0),  # 9: forward left + boost
    (-1.0,  0.0, 0, 0, 0),  # 10: reverse
    (-1.0,  1.0, 0, 0, 0),  # 11: reverse right
    (-1.0, -1.0, 0, 0, 0),  # 12: reverse left
    ( 0.0,  1.0, 0, 0, 0),  # 13: turn right in place
    ( 0.0, -1.0, 0, 0, 0),  # 14: turn left in place
    ( 0.5,  0.0, 0, 1, 0),  # 15: powerslide straight
    ( 0.5,  1.0, 0, 1, 0),  # 16: powerslide hard right
    ( 0.5, -1.0, 0, 1, 0),  # 17: powerslide hard left
    ( 1.0,  0.0, 0, 0, 1),  # 18: jump
]

def action_to_controls(a: int) -> SimpleControllerState:
    t, s, b, hb, j = LUT[a % len(LUT)]
    c = SimpleControllerState()
    c.throttle = t
    c.steer    = s
    c.boost    = bool(b)
    c.handbrake= bool(hb)
    c.jump     = bool(j)
    return c

class MyRLGymBot(BaseAgent):
    def initialize_agent(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model  = TinyPolicy().to(self.device).eval()
        ckpt = latest_checkpoint()
        if ckpt and os.path.isfile(ckpt):
            try:
                state = torch.load(ckpt, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(state, dict) and "state_dict" in state:
                    state_dict = state["state_dict"]
                elif isinstance(state, dict):
                    state_dict = state
                else:
                    state_dict = state
                
                # Fix key naming mismatch: RLGym-PPO uses "model.X" but TinyPolicy uses "net.X"
                fixed_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("model."):
                        new_key = key.replace("model.", "net.")
                        fixed_state_dict[new_key] = value
                    else:
                        fixed_state_dict[key] = value
                
                # Check architecture compatibility before loading
                input_size = 0
                output_size = 0
                if 'net.0.weight' in fixed_state_dict:
                    input_size = fixed_state_dict['net.0.weight'].shape[1]
                    
                    # Find the output layer (final layer with weight)
                    weight_keys = [k for k in fixed_state_dict.keys() if k.endswith('.weight')]
                    if weight_keys:
                        final_layer_key = sorted(weight_keys)[-1]  # Get the highest numbered layer
                        output_size = fixed_state_dict[final_layer_key].shape[0]
                    
                    if input_size != OBS_SIZE or output_size != N_ACTIONS:
                        print(f"[MyRLGymBot] Skipping incompatible checkpoint: {input_size} obs/{output_size} actions, expected {OBS_SIZE}/{N_ACTIONS}")
                        print(f"[MyRLGymBot] Run training with --preset simple to create compatible checkpoint")
                        print(f"[MyRLGymBot] No checkpoint found; running untrained.")
                        return
                
                self.model.load_state_dict(fixed_state_dict)
                
                # Identify training type from path
                training_type = "unknown"
                if "current_aggressive" in ckpt:
                    training_type = "aggressive (ball-chasing focused)"
                elif "current_enhanced" in ckpt:
                    training_type = "enhanced (64-obs game awareness)"
                elif "current_simple" in ckpt:
                    training_type = "simple (32-obs basic)"
                
                print(f"[MyRLGymBot] ✅ Loaded {training_type} checkpoint: {ckpt}")
                print(f"[MyRLGymBot] Architecture: {input_size} observations, {output_size} actions")
            except Exception as e:
                print(f"[MyRLGymBot] Failed to load {ckpt}: {e}")
                print(f"[MyRLGymBot] No checkpoint found; running untrained.")
        else:
            print("[MyRLGymBot] No checkpoint found; running untrained.")

    def get_output(self, game_tick_packet: GameTickPacket) -> SimpleControllerState:
        index = self.index if self.index is not None else 0
        obs = packet_to_obs(game_tick_packet, index)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(obs).unsqueeze(0).to(self.device))
            action = int(torch.argmax(logits, dim=1).item())
        return action_to_controls(action)

def create_agent(config, team, index):
    # RLBot calls this
    return MyRLGymBot(name="MyRLGymBot", team=team, index=index)
