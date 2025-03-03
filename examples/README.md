# Catanatron PyO3 Examples

This directory contains example scripts demonstrating how to use the Catanatron Rust implementation from Python via PyO3 bindings.

## Getting Started

Before running the examples, ensure you have built and installed the Rust implementation:

```bash
cd catanatron_rust
maturin develop
```

This will compile the Rust code and make it available for import in your Python environment.

## Available Examples

### Enhanced PyRust Interface (`enhanced_pyrust_interface.py`)

This example demonstrates the full capabilities of the Rust-Python integration:

```bash
python enhanced_pyrust_interface.py
```

To inspect the game state in detail:

```bash
python enhanced_pyrust_interface.py --inspect
```

This example shows:
- Creating a custom Python player by subclassing the Rust `Player` class
- Running a complete game with Rust backend and Python players
- Accessing and inspecting the game state 
- Using observers to track game progress
- Interacting with the enhanced state representation

## Developing Custom Players

You can develop your own strategy by subclassing the Rust `Player` class and implementing the `decide` method:

```python
from catanatron_rust import Player, Action

class MyCustomPlayer(Player):
    def __init__(self, color):
        super().__init__(color, "MyCustomStrategy")
    
    def decide(self, game_state, playable_actions):
        # Your custom logic here
        # The game_state is a dictionary containing the current state:
        #   - current_player: The current player's color
        #   - current_player_color: Color name (RED, BLUE, etc.)
        #   - is_initial_build_phase: Boolean indicating initial phase
        #   - action_prompt: Current action being requested
        #   - players: Dictionary of player states
        #   - bank: Dictionary of available resources
        #   - buildable_nodes: List of buildable node locations
        #   - buildable_edges: List of buildable edge locations
        
        # Example: choose the first action
        return playable_actions[0]
```

## State Dictionary Structure

The enhanced state representation provides the following information:

```python
{
    # Basic game state
    'current_player': 0,                # Player index (0-3)
    'current_player_color': 'RED',      # Player color name
    'is_initial_build_phase': True,     # Whether in initial build phase
    'current_tick_seat': 0,             # Current player's seat 
    'action_prompt': 'BUILD_SETTLEMENT', # Current action being prompted
    'has_rolled': False,                # Whether current player has rolled
    
    # Bank information
    'bank': {
        'wood': 19,
        'brick': 19,
        'sheep': 19,
        'wheat': 19,
        'ore': 19
    },
    
    # Robber information
    'robber_tile': 7,                   # Current robber tile
    
    # Player information
    'players': {
        'RED': {
            'resources': {'wood': 0, 'brick': 0, ...},
            'development_cards': [],
            'settlements': [0, 3],      # Node IDs of settlements
            'cities': [],               # Node IDs of cities
            'roads': [0, 5]             # Edge IDs of roads
        },
        'BLUE': {...},
        ...
    },
    
    # Board state
    'buildable_nodes': [5, 7, 9, ...],  # Node IDs buildable by current player
    'buildable_edges': [8, 10, 12, ...] # Edge IDs buildable by current player
}
```

## Tips for Implementation

1. **Random Strategies**: For simple random strategies, just select a random action from `playable_actions`
2. **Resource-based Strategies**: Examine the 'players' dictionary to see what resources you have
3. **Expansion Strategies**: Look at 'buildable_nodes' and 'buildable_edges' to plan expansion
4. **Defensive Strategies**: Consider the state of other players to block their expansion

## Performance Considerations

The PyO3 bindings provide excellent performance while maintaining the flexibility of Python:

- The core game logic runs in Rust
- Player decision-making can be implemented in Python
- State conversions are optimized to minimize overhead

For extremely performance-critical applications, consider implementing your players entirely in Rust. 