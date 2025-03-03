"""
A simple script to test the Rust backend directly without relying on catanatron modules.
This can be used to verify that the Rust bindings are working properly.
"""

import os
import sys
import time
import inspect

# Add the Rust module path to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUST_DIR = os.path.join(BASE_DIR, 'catanatron_rust')
sys.path.insert(0, RUST_DIR)

try:
    # Try to import the Rust module directly
    from catanatron_rust import Game as RustGame
    print("✓ Successfully imported Rust backend!")
    
    # Add code to inspect the RustGame constructor
    print("\nInspecting RustGame constructor:")
    
    # Get the signature of the RustGame constructor
    print(f"RustGame class type: {type(RustGame)}")
    
    # Try to get constructor info
    try:
        signature = inspect.signature(RustGame.__init__)
        print(f"Constructor signature: {signature}")
    except (TypeError, ValueError) as e:
        print(f"Could not get constructor signature: {e}")
    
    # Print the doc string if available
    if hasattr(RustGame, '__doc__') and RustGame.__doc__:
        print(f"RustGame docstring: {RustGame.__doc__}")
    else:
        print("No docstring available")
    
    # Try to create an instance with various parameter combinations
    print("\nTrying different constructor calls:")
    
    # Approach 1: Just players
    try:
        players = [SimplePlayer(0), SimplePlayer(1)]
        game = RustGame(players)
        print("✓ RustGame(players) works")
    except Exception as e:
        print(f"✗ RustGame(players) failed: {e}")
    
    # Approach 2: Players and seed
    try:
        players = [SimplePlayer(0), SimplePlayer(1)]
        game = RustGame(players, 42)
        print("✓ RustGame(players, seed) works")
    except Exception as e:
        print(f"✗ RustGame(players, seed) failed: {e}")
    
    # Approach 3: Players, seed, and other params
    try:
        players = [SimplePlayer(0), SimplePlayer(1)]
        game = RustGame(players, 42, 7, 10)
        print("✓ RustGame(players, seed, discard_limit, vps_to_win) works")
    except Exception as e:
        print(f"✗ RustGame(players, seed, discard_limit, vps_to_win) failed: {e}")
    
except ImportError as e:
    print(f"✗ Could not import RustGame: {e}")
    sys.exit(1)

# Define a simple Python player class to use with Rust backend
class SimplePlayer:
    def __init__(self, color):
        self.color = color
        # Add a numeric color value for Rust
        if isinstance(color, int):
            self._rust_color = color
        # For debugging
        print(f"Created player with color={color}, _rust_color={getattr(self, '_rust_color', None)}")
    
    def decide(self, game_state, playable_actions):
        """Choose a random action from the playable actions"""
        import random
        return random.choice(playable_actions)

def run_simple_test():
    """Run a simple test with the Rust backend using our custom player"""
    print("\nRunning simple test with Rust backend...")
    
    # Create players with colors 0 and 1 (RED and BLUE in Rust)
    players = [SimplePlayer(0), SimplePlayer(1)]
    
    try:
        # Create and run a game
        print("Creating game...")
        game = RustGame(players)
        
        print("Playing game...")
        start_time = time.time()
        winner = game.play()
        duration = time.time() - start_time
        
        print(f"Game completed in {duration:.2f} seconds")
        print(f"Winner: Player {winner}")
        return True
    except Exception as e:
        print(f"Error during game: {e}")
        return False

if __name__ == "__main__":
    run_simple_test() 