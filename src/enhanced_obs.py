"""
Enhanced observation builder based on Nexto's sophisticated approach
Provides normalized, multi-player observations with boost locations and timers
"""
import math
import numpy as np
from typing import Any, List
from rlgym_compat import GameState, PlayerData, common_values, BLUE_TEAM, ORANGE_TEAM

# Boost pad locations (same as Nexto)
BOOST_LOCATIONS = np.array([
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (- 940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (- 940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
], dtype=np.float32)

# Determine boost types (large vs small)
BOOST_TYPES = BOOST_LOCATIONS[:, 2] > 72  # Large boosts are higher


def rotation_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (same as Nexto)"""
    trace = np.trace(m)
    q = np.zeros(4)

    if trace > 0:
        s = (trace + 1) ** 0.5
        q[0] = s * 0.5
        s = 0.5 / s
        q[1] = (m[2, 1] - m[1, 2]) * s
        q[2] = (m[0, 2] - m[2, 0]) * s
        q[3] = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = (1 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = 0.5 * s
            q[2] = (m[1, 0] + m[0, 1]) * inv_s
            q[3] = (m[2, 0] + m[0, 2]) * inv_s
            q[0] = (m[2, 1] - m[1, 2]) * inv_s
        elif m[1, 1] > m[2, 2]:
            s = (1 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 1] + m[1, 0]) * inv_s
            q[2] = 0.5 * s
            q[3] = (m[1, 2] + m[2, 1]) * inv_s
            q[0] = (m[0, 2] - m[2, 0]) * inv_s
        else:
            s = (1 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5
            inv_s = 0.5 / s
            q[1] = (m[0, 2] + m[2, 0]) * inv_s
            q[2] = (m[1, 2] + m[2, 1]) * inv_s
            q[3] = 0.5 * s
            q[0] = (m[1, 0] - m[0, 1]) * inv_s

    return -q


class EnhancedObsBuilder:
    """
    Enhanced observation builder based on Nexto's approach
    Provides normalized, team-aware observations with boost and demo tracking
    """
    
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi, tick_skip=8):
        """
        Enhanced observation builder with Nexto's normalization approach
        """
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef  
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.tick_skip = tick_skip
        
        # Normalization factors (from Nexto)
        self._norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)
        self._invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
        
        # Tracking variables
        self.demo_timers = None
        self.boost_timers = None
        
    def reset(self, initial_state: GameState):
        """Reset internal state tracking"""
        if self.demo_timers is None:
            self.demo_timers = np.zeros(len(initial_state.players))
            self.boost_timers = np.zeros(len(initial_state.boost_pads))
        
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        """
        Build enhanced observation based on Nexto's approach
        Returns normalized observation vector
        """
        
        # Initialize tracking if needed
        if self.demo_timers is None:
            self.reset(state)
        
        # Determine if we need inverted view (orange team)
        inverted = player.team_num == ORANGE_TEAM
        
        # Start building observation
        obs = []
        
        # Ball information (3 + 3 + 3 = 9 features)
        if inverted:
            ball = state.inverted_ball
            boost_pads = state.inverted_boost_pads
        else:
            ball = state.ball
            boost_pads = state.boost_pads
        
        # Handle case where ball might not be initialized
        if ball is not None:
            obs.extend(ball.position * self.POS_COEF)  # 3
            obs.extend(ball.linear_velocity * self.LIN_VEL_COEF)  # 3
            obs.extend(ball.angular_velocity * self.ANG_VEL_COEF)  # 3
        else:
            obs.extend([0.0, 0.0, 0.0])  # 3 - default position
            obs.extend([0.0, 0.0, 0.0])  # 3 - default velocity
            obs.extend([0.0, 0.0, 0.0])  # 3 - default angular velocity
        
        # Previous action (8 features)
        obs.extend(previous_action)  # 8
        
        # Boost pad states (34 features)
        obs.extend(boost_pads.astype(float))  # 34
        
        # Player information
        # Self first
        self._add_player_to_obs(obs, player, inverted, is_self=True)
        
        # Teammates and opponents
        teammates = [p for p in state.players if p.team_num == player.team_num and p.car_id != player.car_id]
        opponents = [p for p in state.players if p.team_num != player.team_num]
        
        # Sort by distance to ball for consistent ordering
        if ball is not None:
            ball_pos = ball.position
            teammates.sort(key=lambda p: np.linalg.norm(
                (p.inverted_car_data if inverted else p.car_data).position - ball_pos
            ))
            opponents.sort(key=lambda p: np.linalg.norm(
                (p.inverted_car_data if inverted else p.car_data).position - ball_pos
            ))
        else:
            # Default sorting by car_id if ball position unavailable
            teammates.sort(key=lambda p: p.car_id)
            opponents.sort(key=lambda p: p.car_id)
        
        # Add teammates
        for teammate in teammates:
            self._add_player_to_obs(obs, teammate, inverted, is_self=False)
        
        # Add opponents  
        for opponent in opponents:
            self._add_player_to_obs(obs, opponent, inverted, is_self=False)
        
        # Convert to numpy array and normalize
        obs_array = np.array(obs, dtype=np.float32)
        
        return obs_array
    
    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool, is_self: bool = False):
        """Add player information to observation (based on Nexto's approach)"""
        
        # Get appropriate car data
        if inverted:
            car_data = player.inverted_car_data
        else:
            car_data = player.car_data
        
        # Entity type indicator (5 features)
        obs.extend([
            1.0 if is_self else 0.0,  # IS_SELF
            1.0 if player.team_num == (ORANGE_TEAM if inverted else BLUE_TEAM) and not is_self else 0.0,  # IS_MATE
            1.0 if player.team_num != (ORANGE_TEAM if inverted else BLUE_TEAM) else 0.0,  # IS_OPP
            0.0,  # IS_BALL
            0.0,  # IS_BOOST
        ])
        
        # Position (3 features)
        obs.extend(car_data.position * self.POS_COEF)
        
        # Velocity (3 features)  
        obs.extend(car_data.linear_velocity * self.LIN_VEL_COEF)
        
        # Forward vector (3 features)
        obs.extend(car_data.forward())
        
        # Up vector (3 features)
        obs.extend(car_data.up())
        
        # Angular velocity (3 features)
        obs.extend(car_data.angular_velocity * self.ANG_VEL_COEF)
        
        # Player state (4 features)
        obs.extend([
            player.boost_amount,  # 0-1 normalized
            1.0 if player.is_demoed else 0.0,
            1.0 if player.on_ground else 0.0,
            1.0 if player.has_flip else 0.0,
        ])
        
        # Actions will be added separately if needed (8 features)
        obs.extend([0.0] * 8)  # Placeholder for actions
    
    def get_obs_size(self, max_players=6):
        """Calculate observation size"""
        # Ball: 9, Previous action: 8, Boost pads: 34
        base_size = 9 + 8 + 34
        
        # Each player: type(5) + pos(3) + vel(3) + forward(3) + up(3) + ang_vel(3) + state(4) + actions(8) = 32
        player_size = 32 * max_players
        
        return base_size + player_size


def packet_to_enhanced_obs(packet, index: int, previous_action: np.ndarray) -> np.ndarray:
    """
    Convert RLBot packet to enhanced observation
    Simplified direct conversion without GameState dependency
    """
    # For now, return a simplified observation that matches the expected size
    # This is a temporary fix until we can properly integrate the RLGym GameState
    obs_size = 243  # Enhanced observation size
    
    # Create a basic observation from the packet data
    obs = np.zeros(obs_size, dtype=np.float32)
    
    if index < len(packet.game_cars) and packet.game_cars[index].physics.location.x != 0:
        # Basic ball information (9 features)
        ball = packet.game_ball.physics
        obs[0] = ball.location.x / 4096.0  # Normalized position
        obs[1] = ball.location.y / 5120.0
        obs[2] = ball.location.z / 2044.0
        obs[3] = ball.velocity.x / 2300.0  # Normalized velocity
        obs[4] = ball.velocity.y / 2300.0
        obs[5] = ball.velocity.z / 2300.0
        obs[6] = ball.angular_velocity.x / 6.0  # Normalized angular velocity
        obs[7] = ball.angular_velocity.y / 6.0
        obs[8] = ball.angular_velocity.z / 6.0
        
        # Previous action (8 features starting at index 9)
        if len(previous_action) >= 8:
            obs[9:17] = previous_action[:8]
        
        # Player information (starting at index 17)
        for i, car in enumerate(packet.game_cars[:6]):  # Max 6 players
            base_idx = 17 + i * 32
            if base_idx + 32 <= obs_size:
                # Basic car data (simplified)
                obs[base_idx:base_idx+3] = [car.physics.location.x/4096.0, car.physics.location.y/5120.0, car.physics.location.z/2044.0]
                obs[base_idx+3:base_idx+6] = [car.physics.velocity.x/2300.0, car.physics.velocity.y/2300.0, car.physics.velocity.z/2300.0]
                obs[base_idx+6] = car.boost / 100.0
                obs[base_idx+7] = 1.0 if car.jumped else 0.0
                obs[base_idx+8] = 1.0 if car.double_jumped else 0.0
    
    return obs