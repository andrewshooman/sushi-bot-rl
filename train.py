# main_script.py
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np

from rlgym.api import RLGym, RewardFunction, AgentID
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league import common_values
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    GoalCondition, NoTouchTimeoutCondition, TimeoutCondition, AnyCondition
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    MutatorSequence, FixedTeamSizeMutator, KickoffMutator
)
from rlgym_ppo import Learner
from rlgym_ppo.util import RLGymV2GymWrapper


# ----------------- Components from RLGymExampleBot -----------------
from rlgym.api import ObsBuilder, ActionParser
import math

class ExampleBotObs(ObsBuilder):
    """
    Observation builder based on the working RLGymExampleBot-main implementation
    """
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi):
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_obs_space(self, agent) -> Any:
        # Return tuple format: (type, shape)
        # Our observation has 116 features (as measured from actual output)
        return ('box', (116,))

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        obs_dict = {}
        
        for agent_id in agents:
            # Find the player for this agent
            player = None
            for car_id, car in state.cars.items():
                if car_id == agent_id:
                    player = car
                    break
            
            if player is None:
                # Fallback empty observation
                obs_dict[agent_id] = np.zeros(107, dtype=np.float32)  # Standard size
                continue
            
            # Use team-appropriate physics (inverted for orange team)
            if player.is_orange:
                ball = state.inverted_ball
                # Use inverted boost pads if available
                boost_pads = getattr(state, 'inverted_boost_pads', [0.0] * 34)
            else:
                ball = state.ball
                boost_pads = getattr(state, 'boost_pads', [0.0] * 34)
            
            # Build observation similar to the working example
            obs = []
            
            # Ball info (6 features)
            obs.extend(ball.position * self.POS_COEF)  # 3
            obs.extend(ball.linear_velocity * self.LIN_VEL_COEF)  # 3
            obs.extend(ball.angular_velocity * self.ANG_VEL_COEF)  # 3
            
            # Previous action (8 features) - use zeros for now
            obs.extend([0.0] * 8)  # 8
            
            # Boost pads (34 features) - small + large boost pads
            if hasattr(boost_pads, '__len__'):
                obs.extend(boost_pads[:34])  # 34
            else:
                obs.extend([0.0] * 34)  # 34
            
            # Player info - self
            self._add_player_to_obs(obs, player, player.is_orange, state)
            
            # Add teammates and opponents (simplified - just pad with zeros for now)
            # In a real implementation, we'd iterate through other players
            obs.extend([0.0] * 46)  # Placeholder for other players
            
            obs_dict[agent_id] = np.array(obs, dtype=np.float32)
        
        return obs_dict
    
    def _add_player_to_obs(self, obs: List, player, inverted: bool, state: GameState):
        """Add player data to observation"""
        player_physics = player.inverted_physics if inverted else player.physics
        
        # Position, forward, up vectors, velocities, boost, on_ground, has_flip, is_demoed
        obs.extend(player_physics.position * self.POS_COEF)  # 3
        
        # Calculate forward and up vectors from euler angles
        pitch, yaw, roll = player_physics.euler_angles
        forward_x = math.cos(pitch) * math.cos(yaw)
        forward_y = math.cos(pitch) * math.sin(yaw)
        forward_z = math.sin(pitch)
        obs.extend([forward_x, forward_y, forward_z])  # 3
        
        up_x = -math.sin(roll) * math.sin(pitch) * math.cos(yaw) - math.cos(roll) * math.sin(yaw)
        up_y = -math.sin(roll) * math.sin(pitch) * math.sin(yaw) + math.cos(roll) * math.cos(yaw)
        up_z = math.sin(roll) * math.cos(pitch)
        obs.extend([up_x, up_y, up_z])  # 3
        
        obs.extend(player_physics.linear_velocity * self.LIN_VEL_COEF)  # 3
        obs.extend(player_physics.angular_velocity * self.ANG_VEL_COEF)  # 3
        
        obs.extend([
            player.boost_amount / 100.0,  # 1
            1.0 if player.on_ground else 0.0,  # 1
            1.0 if player.has_flip else 0.0,  # 1
            0.0  # is_demoed placeholder # 1
        ])  # 4


class ExampleBotAction(ActionParser):
    """
    Continuous action parser based on RLGymExampleBot implementation
    """
    def __init__(self):
        pass

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_action_space(self, agent) -> Any:
        # Return tuple format: (type, space_definition)
        # 8 continuous actions: throttle, steer, pitch, yaw, roll, jump, boost, handbrake
        return ('continuous', 8)

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed_actions = {}
        
        for agent_id, action in actions.items():
            if isinstance(action, (int, float)):
                # Convert single value to array
                action = np.array([action], dtype=np.float32)
            elif len(action.shape) == 0:
                action = np.array([float(action)], dtype=np.float32)
            
            # Ensure we have 8 actions
            if len(action) < 8:
                # Pad with zeros
                padded_action = np.zeros(8, dtype=np.float32)
                padded_action[:len(action)] = action
                action = padded_action
            elif len(action) > 8:
                # Truncate
                action = action[:8]
            
            # Clamp first 5 actions to [-1, 1], convert last 3 to binary
            action = action.astype(np.float32)
            action[:5] = np.clip(action[:5], -1, 1)
            action[5:] = (action[5:] > 0).astype(np.float32)
            
            parsed_actions[agent_id] = action
        
        return parsed_actions


# ----------------- Custom rewards -----------------
class SpeedTowardBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        for agent in agents:
            car = state.cars[agent]
            # use team-inverted physics so "toward ball" is always opponent-ward
            car_physics = car.physics if car.is_orange else car.inverted_physics
            ball_physics = state.ball if car.is_orange else state.inverted_ball
            pos_diff = ball_physics.position - car_physics.position
            dist = float(np.linalg.norm(pos_diff) + 1e-6)
            dir_to_ball = pos_diff / dist
            vel = car_physics.linear_velocity
            speed_toward = float(np.dot(vel, dir_to_ball))
            rewards[agent] = max(speed_toward / common_values.CAR_MAX_SPEED, 0.0)
        return rewards


class InAirReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {agent: float(not state.cars[agent].on_ground) for agent in agents}


class VelocityBallToGoalReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        ball = state.ball
        for agent in agents:
            car = state.cars[agent]
            goal_y = -common_values.BACK_NET_Y if car.is_orange else common_values.BACK_NET_Y
            to_goal = np.array([0.0, goal_y, 0.0], dtype=np.float32) - ball.position
            dist = float(np.linalg.norm(to_goal) + 1e-6)
            dir_to_goal = to_goal / dist
            vel_toward = float(np.dot(ball.linear_velocity, dir_to_goal))
            rewards[agent] = max(vel_toward / common_values.BALL_MAX_SPEED, 0.0)
        return rewards


class BallContactReward(RewardFunction[AgentID, GameState, float]):
    """Reward for touching the ball - encourages aggressive play"""
    
    def __init__(self):
        self.last_ball_touch = {}
        self.last_ball_position = None
        self.last_ball_velocity = None
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.last_ball_touch = {}
        self.last_ball_position = initial_state.ball.position.copy()
        self.last_ball_velocity = initial_state.ball.linear_velocity.copy()
    
    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        
        # Detect ball contact by significant velocity change
        current_ball_velocity = state.ball.linear_velocity
        if self.last_ball_velocity is not None:
            velocity_change = np.linalg.norm(current_ball_velocity - self.last_ball_velocity)
            
            # If ball velocity changed significantly, find closest car
            if velocity_change > 100.0:  # Threshold for ball contact
                closest_agent = None
                min_distance = float('inf')
                
                for agent in agents:
                    car = state.cars[agent]
                    distance = float(np.linalg.norm(state.ball.position - car.physics.position))
                    if distance < min_distance:
                        min_distance = distance
                        closest_agent = agent
                
                # Reward the closest car (likely the one that hit the ball)
                for agent in agents:
                    if agent == closest_agent and min_distance < 200.0:  # Close enough to have hit ball
                        rewards[agent] = 1.0
                    else:
                        rewards[agent] = 0.0
            else:
                for agent in agents:
                    rewards[agent] = 0.0
        else:
            for agent in agents:
                rewards[agent] = 0.0
        
        self.last_ball_velocity = current_ball_velocity.copy()
        return rewards


class DistanceToBallReward(RewardFunction[AgentID, GameState, float]):
    """Reward for being close to the ball - encourages ball chasing"""
    
    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass
    
    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        rewards: Dict[AgentID, float] = {}
        
        for agent in agents:
            car = state.cars[agent]
            
            # Calculate distance to ball
            ball_pos = state.ball.position
            car_pos = car.physics.position
            
            distance = float(np.linalg.norm(ball_pos - car_pos))
            
            # Reward inverse distance (closer = higher reward)
            # Max distance on field is about 10000, so normalize
            max_distance = 10000.0
            normalized_distance = min(distance / max_distance, 1.0)
            rewards[agent] = 1.0 - normalized_distance
            
        return rewards


# ----------------- Env builder -----------------
def build_rlgym_v2_env(preset: str = "standard") -> RLGymV2GymWrapper:
    import numpy as _np

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0

    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    # Choose observation builder and action parser based on preset
    if preset == "simple":
        # Simple bot-compatible setup (32 obs, 9 actions)
        obs_builder = SimpleBotObs(obs_size=32)
        action_parser = RepeatAction(SimpleBotAction(), repeats=action_repeat)
    elif preset in ["enhanced", "aggressive"]:
        # Enhanced setup with more game awareness (64 obs, 9 actions)
        obs_builder = SimpleBotObs(obs_size=64)
        action_parser = RepeatAction(SimpleBotAction(), repeats=action_repeat)
    else:
        # Complex setup for research/advanced performance (116 obs, 8 actions)
        obs_builder = ExampleBotObs(
            pos_coef=1.0 / 2300.0,  # Match the working example
            ang_coef=1 / math.pi,
            lin_vel_coef=1.0 / 2300.0,
            ang_vel_coef=1 / math.pi,
        )
        action_parser = RepeatAction(ExampleBotAction(), repeats=action_repeat)
    
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    # Choose reward function based on preset
    if preset == "aggressive":
        reward_fn = CombinedReward(
            # Aggressive ball interaction rewards
            (BallContactReward(), 5.0),                 # Big reward for hitting ball
            (DistanceToBallReward(), 0.5),              # Reward for staying close to ball
            (SpeedTowardBallReward(), 0.5),             # Reward for chasing ball
            (VelocityBallToGoalReward(), 1.0),          # Reward for ball toward goal
            
            # Secondary rewards
            (InAirReward(), 0.001),                     # Reduced aerial emphasis
            (GoalReward(), 10.0),                       # High reward for goals
        )
    else:
        # Standard reward function for other presets
        reward_fn = CombinedReward(
            (InAirReward(), 0.002),
            (SpeedTowardBallReward(), 0.01),
            (VelocityBallToGoalReward(), 0.1),
            (GoalReward(), 10.0),
        )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )

    return RLGymV2GymWrapper(rlgym_env)


def visualize_env(env, n_steps: int = 1000, fps: int = 30):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    from rlgym.rocket_league.api import GameState

    # Reset once and keep the returned obs_dict
    obs_dict, _ = env.reset()

    fig, ax = plt.subplots()
    ax.set_title("Top-down visualization (blue vs orange)")
    ax.set_xlim(-4200, 4200)
    ax.set_ylim(-5400, 5400)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5)

    blue_scatter = ax.plot([], [], marker="o", markersize=8, linestyle="None")[0]
    orange_scatter = ax.plot([], [], marker="o", markersize=8, linestyle="None")[0]
    ball_scatter = ax.plot([], [], marker="x", markersize=8, linestyle="None")[0]
    text_step = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    rng = np.random.default_rng(0)

    def extract_xy(gs: GameState):
        bx, by = gs.ball.position[0], gs.ball.position[1]
        blues_x, blues_y, oranges_x, oranges_y = [], [], [], []
        for _, car in gs.cars.items():
            x, y = car.physics.position[0], car.physics.position[1]
            (oranges_x if car.is_orange else blues_x).append(x)
            (oranges_y if car.is_orange else blues_y).append(y)
        return (np.array(blues_x), np.array(blues_y)), (np.array(oranges_x), np.array(oranges_y)), np.array([bx, by])

    def update(frame_idx):
        nonlocal obs_dict

        # Number of controlled agents = length of MultiDiscrete.nvec
        try:
            n_agents = len(env.action_space.nvec)
        except AttributeError:
            # fallback if action_space doesn’t expose nvec
            n_agents = len(obs_dict)

        # random discrete actions (0..18) for each agent; int32 required by wrapper
        actions = rng.integers(low=0, high=19, size=(n_agents,), dtype=np.int32)

        obs_dict, _, terminated, truncated, _ = env.step(actions)

        if terminated or truncated:
            obs_dict, _ = env.reset()

        gs = env.unwrapped.shared_info["state"]

        (bx, by), (ox, oy), ball_xy = extract_xy(gs)
        blue_scatter.set_data(bx, by)
        orange_scatter.set_data(ox, oy)
        ball_scatter.set_data(ball_xy[0], ball_xy[1])
        text_step.set_text(f"step: {frame_idx}")

        return blue_scatter, orange_scatter, ball_scatter, text_step


# ----------------- Simple Bot-Compatible Classes ---------
class SimpleBotObs(ObsBuilder):
    """
    Enhanced observation builder with improved game awareness.
    Provides comprehensive game state information for better decision making.
    """
    def __init__(self, obs_size=64):  # Increased from 32 for more awareness
        self.POS_COEF = 1.0 / 4096.0  # Position normalization
        self.VEL_COEF = 1.0 / 2300.0  # Velocity normalization
        self.ANG_COEF = 1.0 / 3.14159 # Angle normalization
        self.OBS_SIZE = obs_size

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_obs_space(self, agent) -> Any:
        # Return tuple format: (type, shape) - enhanced observation space
        return ('box', (self.OBS_SIZE,))

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        obs_dict = {}
        
        for agent_id in agents:
            # Find the player for this agent
            player = None
            for car_id, car in state.cars.items():
                if car_id == agent_id:
                    player = car
                    break
            
            if player is None:
                # Fallback empty observation
                obs_dict[agent_id] = np.zeros(self.OBS_SIZE, dtype=np.float32)
                continue
            
            # Use team-appropriate physics (inverted for orange team)
            car_physics = player.inverted_physics if player.is_orange else player.physics
            ball_physics = state.inverted_ball if player.is_orange else state.ball
            
            # Build enhanced observation with more game awareness
            obs = []
            
            # === SELF (CAR) INFORMATION === (12 features)
            # Car location (3 features)
            obs.extend(car_physics.position * self.POS_COEF)
            
            # Car velocity (3 features)  
            obs.extend(car_physics.linear_velocity * self.VEL_COEF)
            
            # Car rotation (3 features)
            pitch, yaw, roll = car_physics.euler_angles
            obs.extend([pitch * self.ANG_COEF, yaw * self.ANG_COEF, roll * self.ANG_COEF])
            
            # Car angular velocity (3 features) - NEW: helps with rotation control
            obs.extend(car_physics.angular_velocity * self.ANG_COEF)
            
            # === BALL INFORMATION === (9 features)
            # Ball location (3 features)
            obs.extend(ball_physics.position * self.POS_COEF)
            
            # Ball velocity (3 features)
            obs.extend(ball_physics.linear_velocity * self.VEL_COEF)
            
            # Ball angular velocity (3 features) - NEW: helps predict ball spin
            obs.extend(ball_physics.angular_velocity * self.ANG_COEF)
            
            # === RELATIVE INFORMATION === (6 features)
            # Distance to ball (1 feature) - NEW
            ball_distance = np.linalg.norm(ball_physics.position - car_physics.position)
            obs.append(ball_distance * self.POS_COEF)
            
            # Direction to ball (normalized) (3 features) - NEW
            ball_direction = ball_physics.position - car_physics.position
            ball_dir_norm = np.linalg.norm(ball_direction)
            if ball_dir_norm > 0:
                ball_direction = ball_direction / ball_dir_norm
            obs.extend(ball_direction)
            
            # Relative velocity to ball (2 features) - NEW: speed toward/away from ball
            rel_velocity = ball_physics.linear_velocity - car_physics.linear_velocity
            speed_toward_ball = np.dot(rel_velocity, ball_direction) * self.VEL_COEF if ball_dir_norm > 0 else 0
            obs.append(speed_toward_ball)
            obs.append(np.linalg.norm(rel_velocity) * self.VEL_COEF)  # Total relative speed
            
            # === CAR STATE === (6 features)
            # Basic states (3 features)
            obs.append(1.0 if player.on_ground else 0.0)
            obs.append(player.boost_amount / 100.0)
            obs.append(1.0 if player.has_flip else 0.0)  # NEW: flip availability
            
            # Advanced states (3 features) - NEW
            obs.append(1.0 if player.is_supersonic else 0.0)  # Supersonic state
            obs.append(1.0 if not player.on_ground else 0.0)  # Airborne state
            obs.append(1.0 if not player.has_flip and not player.on_ground else 0.0)  # Used flip in air
            
            # === BOOST PADS === (6 features) - NEW: strategic awareness
            # Find closest small and large boost pads
            closest_small_dist = float('inf')
            closest_large_dist = float('inf')
            
            # Simple boost pad locations (approximate)
            large_boost_positions = [
                np.array([-3584, -2500, 73]), np.array([3584, -2500, 73]),  # Back corners
                np.array([-3584, 2500, 73]), np.array([3584, 2500, 73]),    # Front corners
                np.array([0, -4240, 73]), np.array([0, 4240, 73])           # Mid sides
            ]
            
            for boost_pos in large_boost_positions:
                # Use team-appropriate positions
                if player.is_orange:
                    boost_pos = np.array([-boost_pos[0], -boost_pos[1], boost_pos[2]])
                dist = np.linalg.norm(car_physics.position - boost_pos)
                closest_large_dist = min(closest_large_dist, dist)
            
            # Add boost information (6 features)
            obs.append(closest_large_dist * self.POS_COEF)  # Distance to closest large boost
            obs.append(1.0 if closest_large_dist < 1000 else 0.0)  # Near large boost
            obs.append(0.5)  # Placeholder for small boost distance
            obs.append(0.0)  # Placeholder for small boost availability
            obs.append(player.boost_amount / 33.0)  # Boost as fraction of supersonic threshold
            obs.append(1.0 if player.boost_amount > 50 else 0.0)  # High boost indicator
            
            # === GAME CONTEXT === (remaining features to reach OBS_SIZE)
            # Goals information - NEW
            # Note: RLGym doesn't directly provide goal positions, using field knowledge
            goal_distance = abs(car_physics.position[1]) - 5120  # Distance from goal line
            obs.append(goal_distance * self.POS_COEF)
            
            # Ball height relative to car - NEW: for aerial awareness
            ball_height_rel = (ball_physics.position[2] - car_physics.position[2]) / 2044.0
            obs.append(ball_height_rel)
            
            # Car's forward direction and turning guidance - ENHANCED for better ball chasing
            car_forward = np.array([
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch)
            ])
            
            # Car's right vector (perpendicular to forward) - define once for reuse
            car_right = np.array([
                -np.sin(yaw),
                np.cos(yaw), 
                0
            ])
            
            # How much facing toward ball (dot product)
            facing_ball = np.dot(car_forward, ball_direction) if ball_dir_norm > 0 else 0
            obs.append(facing_ball)
            
            # NEW: Left/Right turning guidance - which direction to turn to face ball
            if ball_dir_norm > 0:
                # Positive = ball is to the right, negative = ball is to the left
                turn_direction = np.dot(car_right, ball_direction)
                obs.append(turn_direction)  # How much to turn right/left
                
                # Turn urgency - how much turning is needed (inverse of facing_ball)
                turn_urgency = 1.0 - abs(facing_ball)
                obs.append(turn_urgency)
                
                # Ball lateral distance (how far left/right ball is)
                lateral_distance = abs(turn_direction) * ball_distance * self.POS_COEF
                obs.append(lateral_distance)
            else:
                obs.extend([0.0, 0.0, 0.0])
            
            # NEW: Predicted ball position for better interception
            # Simple prediction: where will ball be in 0.5 seconds
            predicted_ball_pos = ball_physics.position + (ball_physics.linear_velocity * 0.5)
            predicted_direction = predicted_ball_pos - car_physics.position
            predicted_dir_norm = np.linalg.norm(predicted_direction)
            if predicted_dir_norm > 0:
                predicted_direction = predicted_direction / predicted_dir_norm
                # Facing predicted ball position
                facing_predicted = np.dot(car_forward, predicted_direction)
                obs.append(facing_predicted)
                
                # Turn direction for predicted position
                predicted_turn = np.dot(car_right, predicted_direction)
                obs.append(predicted_turn)
            else:
                obs.extend([0.0, 0.0])
            
            # Pad observation to exact size
            while len(obs) < self.OBS_SIZE:
                obs.append(0.0)
            
            obs_dict[agent_id] = np.array(obs[:self.OBS_SIZE], dtype=np.float32)
        
        return obs_dict


class SimpleBotAction(ActionParser):
    """
    Enhanced discrete action parser with more turning options for better ball-chasing.
    """
    def __init__(self):
        # ENHANCED Action LUT with more turning variations
        self.LUT = [
            ( 1.0,  0.0, 0, 0, 0),  # 0: forward
            ( 1.0,  0.3, 0, 0, 0),  # 1: forward slight right - NEW
            ( 1.0,  0.6, 0, 0, 0),  # 2: forward medium right - NEW
            ( 1.0,  1.0, 0, 0, 0),  # 3: forward hard right - ENHANCED
            ( 1.0, -0.3, 0, 0, 0),  # 4: forward slight left - NEW
            ( 1.0, -0.6, 0, 0, 0),  # 5: forward medium left - NEW  
            ( 1.0, -1.0, 0, 0, 0),  # 6: forward hard left - ENHANCED
            ( 1.0,  0.0, 1, 0, 0),  # 7: forward + boost
            ( 1.0,  0.6, 1, 0, 0),  # 8: forward right + boost - NEW
            ( 1.0, -0.6, 1, 0, 0),  # 9: forward left + boost - NEW
            (-1.0,  0.0, 0, 0, 0),  # 10: reverse
            (-1.0,  1.0, 0, 0, 0),  # 11: reverse right - NEW for repositioning
            (-1.0, -1.0, 0, 0, 0),  # 12: reverse left - NEW for repositioning
            ( 0.0,  1.0, 0, 0, 0),  # 13: turn right in place - NEW
            ( 0.0, -1.0, 0, 0, 0),  # 14: turn left in place - NEW
            ( 0.5,  0.0, 0, 1, 0),  # 15: powerslide straight
            ( 0.5,  1.0, 0, 1, 0),  # 16: powerslide hard right
            ( 0.5, -1.0, 0, 1, 0),  # 17: powerslide hard left
            ( 1.0,  0.0, 0, 0, 1),  # 18: jump
        ]

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_action_space(self, agent) -> Any:
        # Return tuple format: (type, space_definition) - 19 discrete actions with more turning
        return ('discrete', 19)

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, np.ndarray]:
        parsed_actions = {}
        
        for agent_id, action in actions.items():
            if isinstance(action, (int, float)):
                action_idx = int(action) % 9  # Ensure valid action index
            else:
                action_idx = int(action[0]) % 9 if len(action) > 0 else 0
            
            # Convert discrete action to continuous using LUT
            throttle, steer, boost, handbrake, jump = self.LUT[action_idx]
            
            # Convert to the format expected by RLGym (8-dimensional continuous)
            continuous_action = np.array([
                throttle,   # throttle
                steer,      # steer
                0.0,        # pitch (not used in simple actions)
                0.0,        # yaw (not used in simple actions)
                0.0,        # roll (not used in simple actions)
                float(jump),      # jump
                float(boost),     # boost
                float(handbrake)  # handbrake
            ], dtype=np.float32)
            
            parsed_actions[agent_id] = continuous_action
        
        return parsed_actions


# ----------------- Training Presets -----------------
TRAINING_PRESETS = {
    "quick": {
        "timestep_limit": 500_000,
        "ts_per_iteration": 50_000,
        "ppo_batch_size": 50_000,
        "exp_buffer_size": 150_000,
        "ppo_minibatch_size": 25_000,
        "policy_layer_sizes": [256, 256],
        "critic_layer_sizes": [256, 256],
        "description": "Quick training for testing"
    },
    "standard": {
        "timestep_limit": 2_000_000,
        "ts_per_iteration": 100_000,
        "ppo_batch_size": 100_000,
        "exp_buffer_size": 300_000,
        "ppo_minibatch_size": 50_000,
        "policy_layer_sizes": [512, 512, 256, 256],
        "critic_layer_sizes": [512, 512, 256, 256],
        "description": "Standard training configuration"
    },
    "long": {
        "timestep_limit": 10_000_000,
        "ts_per_iteration": 200_000,
        "ppo_batch_size": 200_000,
        "exp_buffer_size": 600_000,
        "ppo_minibatch_size": 100_000,
        "policy_layer_sizes": [512, 512, 512, 256, 256],
        "critic_layer_sizes": [512, 512, 512, 256, 256],
        "description": "Long training for best results"
    },
    "simple": {
        "timestep_limit": 500_000,
        "ts_per_iteration": 50_000,
        "ppo_batch_size": 50_000,
        "exp_buffer_size": 150_000,
        "ppo_minibatch_size": 25_000,
        "policy_layer_sizes": [256, 256],
        "critic_layer_sizes": [256, 256],
        "description": "Simple training compatible with bot architecture (32 obs, 9 actions)"
    },
    "enhanced": {
        "timestep_limit": 1_000_000,
        "ts_per_iteration": 50_000,
        "ppo_batch_size": 50_000,
        "exp_buffer_size": 150_000,
        "ppo_minibatch_size": 25_000,
        "policy_layer_sizes": [512, 256, 256],
        "critic_layer_sizes": [512, 256, 256],
        "description": "Enhanced training with improved game awareness (64 obs, 19 actions)"
    },
    "aggressive": {
        "timestep_limit": 2_000_000,
        "ts_per_iteration": 50_000,
        "ppo_batch_size": 50_000,
        "exp_buffer_size": 150_000,
        "ppo_minibatch_size": 25_000,
        "policy_layer_sizes": [512, 256, 256],
        "critic_layer_sizes": [512, 256, 256],
        "description": "Aggressive ball-chasing training with enhanced rewards for ball contact (64 obs, 19 actions)"
    }
}


# ----------------- Train -----------------
def train(preset: str = "standard", timestep_limit: int | None = None, use_rlviser: bool = False):
    """
    Trains a PPO policy with rlgym_ppo 2.x.
    
    Args:
        preset: Training preset to use ("quick", "standard", "long")
        timestep_limit: Override timestep limit if specified
        use_rlviser: Whether to use RLViser for visualization
    """
    import subprocess
    import os
    
    # Start RLViser if requested
    rlviser_process = None
    if use_rlviser:
        # Try different possible paths for RLViser executable
        possible_paths = [
            os.path.join("rlviser-0.8.2", "target", "release", "rlviser.exe"),
            os.path.join("rlviser-0.8.2", "target", "debug", "rlviser.exe"),
            os.path.join("rlviser-0.8.2", "rlviser.exe")
        ]
        
        rlviser_path = None
        for path in possible_paths:
            if os.path.exists(path):
                rlviser_path = path
                break
        
        if rlviser_path:
            print(f"Starting RLViser from {rlviser_path}...")
            try:
                rlviser_process = subprocess.Popen([rlviser_path], 
                                                 cwd=os.path.dirname(rlviser_path))
                print("RLViser started successfully!")
                print("Note: You can view the training visualization in the RLViser window")
            except Exception as e:
                print(f"Failed to start RLViser: {e}")
                use_rlviser = False
        else:
            print("RLViser executable not found.")
            print("Possible solutions:")
            print("1. Build RLViser: cd rlviser-0.8.2 && cargo build --release")
            print("2. Check if RLViser is properly configured as a binary in Cargo.toml")
            print("3. Training will continue without visualization")
            use_rlviser = False
    
    # Get preset configuration
    if preset not in TRAINING_PRESETS:
        print(f"Unknown preset '{preset}'. Available presets:")
        for name, config in TRAINING_PRESETS.items():
            print(f"  {name}: {config['description']}")
        return
    
    config = TRAINING_PRESETS[preset].copy()
    
    # Override timestep limit if specified
    if timestep_limit is not None:
        config["timestep_limit"] = timestep_limit
    
    print(f"Using preset '{preset}': {config['description']}")
    print(f"Training for {config['timestep_limit']:,} timesteps")
    if use_rlviser:
        print("RLViser visualization enabled")
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join("data", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Use consistent directory name based on preset (replaces older runs)
    import time
    timestamp = int(time.time() * 1000000)  # Microsecond timestamp to avoid conflicts
    run_dir = os.path.join(checkpoint_dir, f"new_{preset}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Checkpoints will be saved to: {run_dir}")
    print(f"Note: This will replace any existing '{preset}' training checkpoints")
    
    n_proc = 8
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    # Create a partial function that can be pickled
    from functools import partial
    env_factory = partial(build_rlgym_v2_env, preset)

    try:
        learner = Learner(
            env_factory,                     # pass the factory as first positional arg for widest compat
            n_proc=n_proc,
            min_inference_size=min_inference_size,
            metrics_logger=None,
            ppo_batch_size=config["ppo_batch_size"],
            policy_layer_sizes=config["policy_layer_sizes"],
            critic_layer_sizes=config["critic_layer_sizes"],
            ts_per_iteration=config["ts_per_iteration"],
            exp_buffer_size=config["exp_buffer_size"],
            ppo_minibatch_size=config["ppo_minibatch_size"],
            ppo_ent_coef=0.01,
            policy_lr=1e-4,
            critic_lr=1e-4,
            ppo_epochs=2,
            standardize_returns=True,
            standardize_obs=False,
            save_every_ts=50_000 if preset in ["simple", "enhanced", "aggressive"] else 1_000_000,  # Save frequently for testing presets
            timestep_limit=config["timestep_limit"],
            log_to_wandb=False,
            checkpoints_save_folder=run_dir,
        )
        learner.learn()
    finally:
        # Clean up RLViser process
        if rlviser_process:
            print("Stopping RLViser...")
            rlviser_process.terminate()
            try:
                rlviser_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rlviser_process.kill()


def convert_model_for_bot(checkpoint_path: str, output_path: str | None = None):
    """
    Convert a trained RLGym-PPO model to the format expected by the bot.
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        output_path: Where to save the converted model (default: same directory as checkpoint)
    """
    import torch
    import os
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(checkpoint_path), "bot_model.pt")
    
    print(f"Converting model from {checkpoint_path} to {output_path}")
    
    try:
        # Load the trained model
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract the policy network (assuming it's stored under 'policy' key)
        if 'policy' in checkpoint:
            policy_state = checkpoint['policy']
        elif 'model' in checkpoint:
            policy_state = checkpoint['model']
        else:
            # Try to find the policy network in the checkpoint
            print("Available keys in checkpoint:", list(checkpoint.keys()))
            return False
        
        # Create the bot's model architecture
        from bot import TinyPolicy
        bot_model = TinyPolicy()
        
        print("WARNING: Model architectures don't match!")
        print("Trained model uses DefaultObs (92 features) -> Large network -> 90 actions")
        print("Bot expects 32 features -> [256, 256] -> 9 actions")
        print("You'll need to either:")
        print("1. Retrain with compatible architecture, or")
        print("2. Create a proper model conversion/adapter")
        
        return False
        
    except Exception as e:
        print(f"Error converting model: {e}")
        return False


# ----------------- Main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RocketLeague RL Training Script")
    
    # Main modes
    parser.add_argument("--train", action="store_true", help="Run PPO training")
    parser.add_argument("--viz", action="store_true", help="Run simple top-down visualization")
    
    # Training options
    parser.add_argument("--preset", type=str, default="standard", 
                       choices=list(TRAINING_PRESETS.keys()),
                       help="Training preset to use (default: standard)")
    parser.add_argument("--steps", type=int, default=None, 
                       help="Override timestep limit for training")
    parser.add_argument("--rlviser", action="store_true", 
                       help="Enable RLViser visualization during training")
    
    # Visualization options
    parser.add_argument("--viz-steps", type=int, default=2000,
                       help="Number of steps for visualization (default: 2000)")
    parser.add_argument("--viz-fps", type=int, default=30,
                       help="FPS for visualization (default: 30)")
    
    # Information
    parser.add_argument("--list-presets", action="store_true",
                       help="List available training presets")
    
    # Model conversion
    parser.add_argument("--convert-model", type=str,
                       help="Convert a trained model checkpoint for use with the bot")
    
    args = parser.parse_args()

    if args.list_presets:
        print("Available training presets:")
        for name, config in TRAINING_PRESETS.items():
            print(f"  {name}: {config['description']}")
            print(f"    - Timesteps: {config['timestep_limit']:,}")
            print(f"    - Network: {config['policy_layer_sizes']}")
            print()
    elif args.convert_model:
        convert_model_for_bot(args.convert_model)
    elif args.train:
        print(f"Starting training with preset '{args.preset}'...")
        if args.steps:
            print(f"Override timestep limit: {args.steps:,}")
        if args.rlviser:
            print("RLViser visualization enabled")
        print()
        train(preset=args.preset, timestep_limit=args.steps, use_rlviser=args.rlviser)
    elif args.viz:
        print("Starting visualization...")
        env = build_rlgym_v2_env("standard")
        visualize_env(env, n_steps=args.viz_steps, fps=args.viz_fps)
    else:
        print("RocketLeague RL Bot Training System")
        print("=====================================")
        print()
        print("Available commands:")
        print("  --train          Start PPO training")
        print("  --viz            Run visualization")
        print("  --list-presets   Show training presets")
        print("  --convert-model  Convert trained model for bot")
        print()
        print("Quick start examples:")
        print("  python train.py --train --preset aggressive")  
        print("  python train.py --train --preset enhanced --steps 500000")
        print("  python train.py --train --preset quick --rlviser")
        print("  python train.py --viz --viz-steps 1000")
        print()
        print("Use --help for detailed options.")
