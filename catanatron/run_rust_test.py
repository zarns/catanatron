import os
import sys
import time
from collections import defaultdict

# Add all necessary paths to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_experimental'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_rust'))

# Now try importing the rust module
try:
    from catanatron_rust import Game as RustGame
    print("Successfully imported Rust backend!")
    RUST_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import Rust backend: {e}")
    RUST_AVAILABLE = False
    sys.exit(1)

# Import other required modules
try:
    # Import from catanatron_core
    from catanatron.players.random import RandomPlayer
    from catanatron.models.player import Color

    print("Successfully imported core modules!")
except ImportError as e:
    print(f"Failed to import core modules: {e}")
    sys.exit(1)

def run_benchmark(num_games=10):
    """Run a benchmark comparing Python and Rust implementations"""
    print(f"Starting benchmark with {num_games} games...")
    
    # Create players
    colors = [Color.RED, Color.BLUE]
    players = [RandomPlayer(colors[i]) for i in range(2)]
    
    # Run games with Rust backend
    print("\nRunning with Rust backend...")
    start_time = time.time()
    
    for i in range(num_games):
        game = RustGame(players, seed=i)
        winner = game.play()
        print(f"Game {i+1}: Winner = {winner}")
    
    rust_duration = time.time() - start_time
    rust_games_per_sec = num_games / rust_duration
    
    print(f"\nBenchmark Results:")
    print(f"Rust: {rust_duration:.2f} seconds total, {rust_games_per_sec:.2f} games/sec")

if __name__ == "__main__":
    num_games = 5
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
        except ValueError:
            pass
    
    run_benchmark(num_games) 