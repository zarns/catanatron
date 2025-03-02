# Catanatron Rust

This is a Rust implementation of the Catanatron game engine with Python bindings.

## Overview

Catanatron Rust provides a high-performance implementation of the Settlers of Catan board game mechanics. 
The project includes:

- A complete Rust implementation of game state and rules
- Python bindings using PyO3 and Maturin
- Support for both pure Rust and Python players
- Benchmarking tools for performance comparison

## Usage

### Building and Installation

```bash
# Build the Rust library
cargo build

# Build and install the Python package
maturin develop
```

### Running Tests

```bash
# Run Rust tests
cargo test

# Run Python integration tests
cd ..
python -m unittest catanatron.test_rust_game
```

### Running the Demo

```bash
cd ..
python -m catanatron.test_rust_game
```

## Python-Rust Integration

### Architecture

The integration between Python and Rust is designed with the following components:

1. **Game Class**: A PyO3-exported Rust class that provides the main interface for Python code
2. **PythonPlayerWrapper**: Allows Python player implementations to be used within the Rust engine
3. **State Conversion**: Converts between Rust and Python representations of game state

### Using Python Players with Rust

Python players must implement a `decide` method that accepts two parameters:

1. `game_state`: A dictionary containing the current game state
2. `playable_actions`: A list of possible actions to choose from

Example Python player:

```python
class SimplePythonPlayer(Player):
    def __init__(self, color):
        self.color = color
        
    def decide(self, game_state, playable_actions):
        # Process game state
        if isinstance(game_state, dict):
            current_color = game_state.get('current_color')
            is_initial = game_state.get('is_initial_build_phase', False)
            print(f"Player {self.color} deciding (current={current_color}, initial={is_initial})")
        
        # Choose an action
        return random.choice(playable_actions)
```

### Game State Dictionary

The Rust engine provides the following information to Python players in the game state dictionary:

- `current_color`: The color of the player whose turn it is
- `action_prompt`: The current action prompt (e.g., BuildInitialSettlement)
- `is_initial_build_phase`: Whether the game is in the initial build phase
- `current_tick_seat`: The current tick seat number
- `resources`: The player's current resources (wood, brick, sheep, wheat, ore)
- `dev_cards`: The player's development cards
- `victory_points`: The player's current victory points
- `seating_order`: The seating order of players

### Action Representation

Actions in Rust are represented as strings like:

```
Action::BuildSettlement { color: 0, node_id: 12 }
```

The Python player should return either:
1. The index of the chosen action in the `playable_actions` list
2. The exact string representation of the chosen action

## Project Structure

- `src/`: Rust source code
  - `lib.rs`: Main library and Python bindings
  - `game.rs`: Game implementation
  - `state.rs`: Game state implementation
  - `players/`: Player implementations
    - `mod.rs`: Player trait and PythonPlayerWrapper
    - `random_player.rs`: RandomPlayer implementation
- `pyproject.toml`: Python package configuration
- `Cargo.toml`: Rust package configuration

## Migration from Python to Rust

This project represents a migration of the Catanatron game engine from Python to Rust, with the following goals:

1. Improve performance
2. Maintain compatibility with existing Python code
3. Support existing Python player implementations
4. Enable gradual migration of game logic

The migration strategy involves:

1. Implementing core game mechanics in Rust
2. Creating Python bindings using PyO3
3. Wrapping Python players to work with Rust
4. Ensuring feature parity with the Python implementation

## Performance Comparison

Initial benchmarks show significant performance improvements:

- Game simulation is ~10-20x faster in Rust
- Memory usage is reduced
- Python players can still be used with minimal overhead

## Future Work

- Complete implementation of all game rules
- Improve state representation
- Enhance Python bindings
- Add more sophisticated Rust player implementations
