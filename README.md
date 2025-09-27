# Sushi Bot RL - Rocket League Reinforcement Learning Bot

A reinforcement learning bot for Rocket League built using RLGym and RLGym-PPO. This project trains intelligent agents to play Rocket League using state-of-the-art PPO (Proximal Policy Optimization) algorithms.

## 🚀 Features

- **Multiple Training Presets**: Quick, standard, and long training configurations
- **Advanced Observation System**: 116-feature observation space with normalized car physics, ball state, and boost information
- **Discrete Action Space**: 19 discrete actions covering all essential Rocket League maneuvers
- **RLViser Integration**: Real-time visualization of training sessions
- **Checkpoint Management**: Automatic saving and loading of training checkpoints
- **Flexible Architecture**: Support for simple, enhanced, and aggressive bot configurations

## 📁 Project Structure

```
sushi-bot-rl/
├── bot.py                 # Main RLBot agent implementation
├── train.py              # Training script with multiple presets
├── test_bot.py           # Bot testing utilities
├── create_checkpoint.py  # Checkpoint creation tools
├── appearance.cfg        # Bot appearance configuration
├── bot.cfg              # RLBot configuration
├── TRAINING_SETUP.md    # Detailed training documentation
├── data/
│   └── checkpoints/     # Saved model checkpoints
│       ├── current_simple/
│       ├── current_enhanced/
│       └── current_aggressive/
└── sample-project/      # Example RLBot project structure
```

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- RLBot framework
- RLGym and RLGym-PPO
- PyTorch
- RocketSim (for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/andrewshooman/sushi-bot-rl.git
cd sushi-bot-rl
```

2. Install dependencies:
```bash
pip install rlbot rlgym rlgym-ppo torch numpy
```

3. Set up RocketSim for training (follow RLGym documentation)

## 🎯 Training

### Quick Start

Train with default enhanced preset:
```bash
python train.py
```

### Training Presets

- **Quick**: `python train.py --preset quick` (500K steps, 4 workers)
- **Standard**: `python train.py --preset standard` (2M steps, 8 workers) 
- **Long**: `python train.py --preset long` (10M steps, 16 workers)

### Custom Training

```bash
# Train for specific number of steps
python train.py --steps 5000000

# Train with RLViser visualization
python train.py --rlviser

# List all available presets
python train.py --list-presets
```

### Training Features

- **Observation Space**: 116 features including:
  - Car physics (position, rotation, velocity, angular velocity)
  - Ball state and prediction
  - Boost information and pad states
  - Team and opponent data

- **Action Space**: 19 discrete actions covering:
  - Throttle, brake, steering
  - Jumping, dodging, air roll
  - Boost usage and powerslide

- **Reward System**: Combined rewards for:
  - Ball touches and possession
  - Goal scoring and saves
  - Boost management
  - Positioning and game sense

## 🤖 Running the Bot

### In RLBot

1. Add the bot to your RLBot match configuration
2. The bot will automatically load the latest checkpoint from `data/checkpoints/`
3. Priority order: `current_aggressive` > `current_enhanced` > `current_simple`

### Testing

```bash
# Test bot functionality
python test_bot.py

# Test training setup
python test_setup.py
```

## 📊 Checkpoint Management

Checkpoints are automatically saved during training:

- **Location**: `data/checkpoints/`
- **Format**: `run_TIMESTAMP_PRESET/UNIQUE_ID/PPO_POLICY.pt`
- **Current Models**: Stored in `current_simple/`, `current_enhanced/`, `current_aggressive/`

The bot automatically loads the most recent compatible checkpoint based on the naming convention.

## 🎮 Bot Configurations

### Simple Bot
- Basic observation space (32 features)
- Simple action space (9 actions)
- Fast inference, good for testing

### Enhanced Bot
- Full observation space (116 features)
- Complete action space (19 actions)
- Balanced performance and complexity

### Aggressive Bot
- Enhanced observations with aggressive reward tuning
- Optimized for offensive play
- Higher risk/reward strategies

## 🔧 Configuration Files

- `bot.cfg`: RLBot agent configuration
- `appearance.cfg`: Visual appearance settings
- Training parameters can be adjusted in `train.py`

## 📈 Performance Monitoring

Training progress is logged with:
- Episode rewards and statistics
- Goal/save rates
- Training loss metrics
- Checkpoint save confirmations

Use RLViser (`--rlviser` flag) for real-time visual monitoring during training.

## 🐛 Troubleshooting

### Common Issues

1. **Checkpoint Loading Errors**: Ensure checkpoint architecture matches bot configuration
2. **Training Crashes**: Check RocketSim installation and system resources
3. **Performance Issues**: Reduce worker count or batch size for lower-end hardware

### Architecture Compatibility

Current training uses:
- 116-feature observations (ExampleBotObs)
- 19-action discrete space
- PPO with large networks

Ensure bot configuration matches training architecture for proper checkpoint loading.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. See the LICENSE file for details.

## 🙏 Acknowledgments

- RLGym team for the excellent Rocket League gym environment
- RLBot community for the bot framework
- RLGym-PPO for the PPO implementation
- RocketSim for fast physics simulation

## ✅ TODO List

### High Priority

- [x] **Anti-Wall-Riding Behavior** ✅ **COMPLETED**
  - ✅ Added GroundContactReward to encourage ground play
  - ✅ Added WallContactPenalty to discourage wall riding  
  - ✅ Removed InAirReward that was encouraging aerial play
  - ✅ Updated both aggressive and standard reward presets

- [ ] **Architecture Compatibility Fix**
  - Align training observation space (116 features) with bot expectations (64 features)
  - Ensure training action space (19 actions) matches bot implementation
  - Update bot.py to handle enhanced observation format properly

- [ ] **Model Performance Validation**
  - Test current checkpoints in actual matches
  - Benchmark performance against baseline bots
  - Validate goal scoring and defensive capabilities

- [ ] **Training Stability**
  - Monitor training convergence across different presets
  - Fix any NaN/infinity issues in reward calculations
  - Optimize hyperparameters for faster convergence

### Medium Priority

- [ ] **Enhanced Reward System**
  - Implement aerial play rewards
  - Add team play and passing rewards
  - Tune reward weights for different bot personalities

- [ ] **Bot Variants**
  - Complete aggressive bot training and testing
  - Implement defensive specialist variant
  - Create 1v1 specialized training preset

- [ ] **Checkpoint Management**
  - Implement checkpoint versioning system
  - Add model comparison utilities
  - Create checkpoint rollback functionality

- [ ] **Documentation**
  - Add detailed training metrics explanation
  - Create bot deployment guide
  - Document reward function components

### Low Priority

- [ ] **Performance Optimization**
  - Profile training bottlenecks
  - Optimize observation building
  - Implement multi-GPU training support

- [ ] **Advanced Features**
  - Add curriculum learning stages
  - Implement self-play training
  - Create tournament bracket system for model evaluation

- [ ] **Code Quality**
  - Add comprehensive unit tests
  - Implement continuous integration
  - Add type hints throughout codebase
  - Refactor duplicate code sections

- [ ] **User Experience**
  - Create GUI for training management
  - Add real-time training metrics dashboard
  - Implement one-click deployment system

### Research & Experimentation

- [ ] **Alternative Algorithms**
  - Experiment with SAC (Soft Actor-Critic)
  - Test IMPALA for distributed training
  - Compare with Rainbow DQN variants

- [ ] **Advanced Observations**
  - Implement visual observations from game camera
  - Add opponent behavior prediction features
  - Experiment with attention mechanisms

- [ ] **Training Environments**
  - Create custom training scenarios
  - Implement dynamic difficulty adjustment
  - Add multi-agent cooperative training

## 📚 Additional Resources

- [RLGym Documentation](https://rlgym.org/)
- [RLBot Framework](https://rlbot.org/)
- [Training Setup Guide](TRAINING_SETUP.md)

---

**Status**: ✅ **ACTIVELY TRAINING** - Anti-wall-riding aggressive preset running (1M steps). Checkpoint system standardized, bot performance improving with ground-focused rewards.