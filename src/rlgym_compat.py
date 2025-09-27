"""
RLGym compatibility layer for the sushi bot
Provides GameState and related structures for better bot implementation
"""
import math
import numpy as np
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo
from rlbot.utils.structures.ball_prediction_struct import BallPrediction


class CarData:
    def __init__(self, car_info: PlayerInfo, inverted=False):
        self.position = np.array([car_info.physics.location.x, 
                                 car_info.physics.location.y, 
                                 car_info.physics.location.z], dtype=np.float32)
        
        self.linear_velocity = np.array([car_info.physics.velocity.x,
                                        car_info.physics.velocity.y,
                                        car_info.physics.velocity.z], dtype=np.float32)
        
        self.angular_velocity = np.array([car_info.physics.angular_velocity.x,
                                         car_info.physics.angular_velocity.y,
                                         car_info.physics.angular_velocity.z], dtype=np.float32)
        
        # Store rotation matrix from RLBot rotation
        rotation = car_info.physics.rotation
        self._rotation_matrix = self._euler_to_rotation_matrix(
            rotation.pitch, rotation.yaw, rotation.roll
        )
        
        if inverted:
            # Invert for orange team perspective
            self.position *= np.array([-1, -1, 1])
            self.linear_velocity *= np.array([-1, -1, 1])
            self.angular_velocity *= np.array([-1, -1, 1])
    
    def forward(self):
        """Get forward vector from rotation matrix"""
        return self._rotation_matrix[:, 0]
    
    def right(self):
        """Get right vector from rotation matrix"""
        return self._rotation_matrix[:, 1]
    
    def up(self):
        """Get up vector from rotation matrix"""
        return self._rotation_matrix[:, 2]
    
    def rotation_mtx(self):
        """Get rotation matrix"""
        return self._rotation_matrix
    
    @staticmethod
    def _euler_to_rotation_matrix(pitch, yaw, roll):
        """Convert euler angles to rotation matrix"""
        CR = math.cos(roll)
        SR = math.sin(roll)
        CP = math.cos(pitch)
        SP = math.sin(pitch)
        CY = math.cos(yaw)
        SY = math.sin(yaw)
        
        matrix = np.array([
            [CP * CY, CP * SY, SP],
            [CY * SP * SR - CR * SY, SY * SP * SR + CR * CY, -CP * SR],
            [-CR * CY * SP - SR * SY, -CR * SY * SP + SR * CY, CP * CR]
        ], dtype=np.float32)
        
        return matrix


class BallData:
    def __init__(self, ball_info, inverted=False):
        self.position = np.array([ball_info.physics.location.x,
                                 ball_info.physics.location.y,
                                 ball_info.physics.location.z], dtype=np.float32)
        
        self.linear_velocity = np.array([ball_info.physics.velocity.x,
                                        ball_info.physics.velocity.y,
                                        ball_info.physics.velocity.z], dtype=np.float32)
        
        self.angular_velocity = np.array([ball_info.physics.angular_velocity.x,
                                         ball_info.physics.angular_velocity.y,
                                         ball_info.physics.angular_velocity.z], dtype=np.float32)
        
        if inverted:
            # Invert for orange team perspective
            self.position *= np.array([-1, -1, 1])
            self.linear_velocity *= np.array([-1, -1, 1])
            self.angular_velocity *= np.array([-1, -1, 1])


class PlayerData:
    def __init__(self, car_info: PlayerInfo, car_id: int, inverted=False):
        self.car_id = car_id
        self.team_num = car_info.team
        self.is_demoed = car_info.is_demolished
        self.on_ground = car_info.has_wheel_contact
        self.ball_touched = car_info.touched_ball
        self.has_flip = car_info.double_jumped == False  # Has flip if hasn't double jumped
        self.boost_amount = car_info.boost / 100.0  # Normalize to 0-1
        
        # Create car data for both normal and inverted views
        self.car_data = CarData(car_info, inverted=False)
        self.inverted_car_data = CarData(car_info, inverted=True)


class GameState:
    def __init__(self, field_info=None):
        self.field_info = field_info
        self.players = []
        self.ball = None
        self.inverted_ball = None
        self.boost_pads = np.zeros(34, dtype=bool)  # Standard 34 boost pads
        self.inverted_boost_pads = np.zeros(34, dtype=bool)
        self.blue_score = 0
        self.orange_score = 0
        
    def decode(self, packet: GameTickPacket, ticks_elapsed: int):
        """Update game state from RLBot packet"""
        # Update scores
        self.blue_score = packet.teams[0].score
        self.orange_score = packet.teams[1].score
        
        # Update ball
        self.ball = BallData(packet.game_ball, inverted=False)
        self.inverted_ball = BallData(packet.game_ball, inverted=True)
        
        # Update boost pads
        for i in range(len(packet.game_boosts)):
            if i < len(self.boost_pads):
                self.boost_pads[i] = packet.game_boosts[i].is_active
                self.inverted_boost_pads[i] = packet.game_boosts[i].is_active
        
        # Update players
        self.players = []
        for i in range(packet.num_cars):
            if packet.game_cars[i].team < 2:  # Valid team
                player = PlayerData(packet.game_cars[i], i)
                self.players.append(player)


# Common values from RLGym
class CommonValues:
    BLUE_TEAM = 0
    ORANGE_TEAM = 1
    
    # Field dimensions
    SIDE_WALL_X = 4096
    BACK_NET_Y = 5120
    CEILING_Z = 2044
    BACK_WALL_Y = 5120
    
    # Physics constants
    CAR_MAX_SPEED = 2300
    BALL_MAX_SPEED = 6000
    BOOST_MAX = 100
    
    # Car dimensions
    CAR_LENGTH = 118
    CAR_WIDTH = 84.2
    CAR_HEIGHT = 36.16

# Create common_values module-like object
common_values = CommonValues()

# Export key constants
BLUE_TEAM = 0
ORANGE_TEAM = 1