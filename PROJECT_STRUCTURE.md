# Project Structure

## Directory Organization

```
sushi-bot-rl/
├── src/                    # Source code
│   ├── bot.py             # Main RLBot agent
│   ├── train.py           # Training script
│   ├── enhanced_obs.py    # Enhanced observation builder
│   ├── enhanced_actions.py # Enhanced action parser
│   ├── rlgym_compat.py    # RLGym compatibility layer
│   ├── bot.cfg            # RLBot configuration
│   └── appearance.cfg     # Bot appearance settings
├── tests/                 # Test files
│   ├── test_bot.py        # Bot functionality tests
│   ├── test_checkpoint_loading.py # Checkpoint loading tests
│   └── test_setup.py      # Setup validation tests
├── tools/                 # Utility scripts
│   ├── create_checkpoint.py # Checkpoint creation utility
│   └── standardize_checkpoints.py # Checkpoint organization tool
├── docs/                  # Documentation
│   ├── ANTI_WALL_RIDING_MODS.md # Anti-wall-riding documentation
│   ├── CHECKPOINT_STANDARDIZATION.md # Checkpoint system docs
│   └── TRAINING_SETUP.md  # Training setup guide
├── examples/              # Example projects
│   ├── sample-project/    # Basic RLBot example
│   └── sample-project-2/  # Advanced examples (Nexto, etc.)
├── data/                  # Data directory
│   └── checkpoints/       # Model checkpoints
│       └── standardized/  # Organized checkpoint format
├── rlviser-0.8.2/        # RLViser visualization tool
├── README.md             # Main project documentation
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup script
└── .gitattributes       # Git configuration
```

## File Descriptions

### Core Source Files (`src/`)
- **`bot.py`**: Main RLBot agent implementation with checkpoint loading
- **`train.py`**: Comprehensive training script with multiple presets
- **`enhanced_obs.py`**: Nexto-inspired observation builder with normalization
- **`enhanced_actions.py`**: Sophisticated action parser with lookup tables
- **`rlgym_compat.py`**: Compatibility layer for RLGym integration

### Test Files (`tests/`)
- **`test_bot.py`**: Unit tests for bot functionality
- **`test_checkpoint_loading.py`**: Tests for checkpoint system
- **`test_setup.py`**: Validation tests for training setup

### Tools (`tools/`)
- **`create_checkpoint.py`**: Creates compatible model checkpoints
- **`standardize_checkpoints.py`**: Organizes checkpoints into standard format

### Documentation (`docs/`)
- **`ANTI_WALL_RIDING_MODS.md`**: Details on wall-riding prevention
- **`CHECKPOINT_STANDARDIZATION.md`**: Checkpoint organization system
- **`TRAINING_SETUP.md`**: Training configuration and troubleshooting

### Examples (`examples/`)
- **`sample-project/`**: Basic RLBot project template
- **`sample-project-2/`**: Advanced bot examples including Nexto

## Import Structure

### Internal Imports
- Files in `src/` can import each other directly
- Tools and tests use `sys.path` manipulation to import from `src/`
- Example: `sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))`

### Package Structure
- Main package: `sushi-bot-rl`
- Source code in `src/` directory
- Installable via `pip install -e .` for development

## Running the Project

### From Source Directory
```bash
cd src/
python train.py --train --preset aggressive
```

### After Installation
```bash
pip install -e .
sushi-bot-train --preset aggressive
```

### Running Tests
```bash
cd tests/
python test_checkpoint_loading.py
```

### Using Tools
```bash
cd tools/
python standardize_checkpoints.py
```