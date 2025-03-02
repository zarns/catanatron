#!/usr/bin/env python
"""
Example script demonstrating how to use the Rust backend directly.
This script compares the performance of the Python and Rust backends
for a simple game simulation.
"""

import time
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from catanatron import RandomPlayer, Color
from catanatron_experimental.rust_bridge import create_game, is_rust_available

def run_performance_comparison(num_games=10):
    """Run a simple performance comparison between Python and Rust backends."""
    print(f"Running performance comparison with {num_games} games")
    
    # Setup players
    players = [
        RandomPlayer(Color.RED),
        RandomPlayer(Color.BLUE),
        RandomPlayer(Color.WHITE),
        RandomPlayer(Color.ORANGE),
    ]
    
    # Run with Python backend
    start_time = time.time()
    python_winners = []
    for _ in range(num_games):
        game = create_game(players, use_rust=False)
        winner = game.play()
        python_winners.append(winner)
    python_time = time.time() - start_time
    
    print(f"Python backend: {python_time:.2f} seconds")
    
    # Check if Rust backend is available
    if not is_rust_available():
        print("Rust backend is not available. Install it with 'cd catanatron_rust && maturin develop'")
        return
    
    # Run with Rust backend
    start_time = time.time()
    rust_winners = []
    for _ in range(num_games):
        game = create_game(players, use_rust=True)
        winner = game.play()
        rust_winners.append(winner)
    rust_time = time.time() - start_time
    
    print(f"Rust backend: {rust_time:.2f} seconds")
    
    # Calculate speedup
    speedup = python_time / rust_time if rust_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results match (should be different due to randomness but distribution should be similar)
    print(f"Python winners distribution: {count_winners(python_winners)}")
    print(f"Rust winners distribution: {count_winners(rust_winners)}")

def count_winners(winners):
    """Count the number of wins for each color."""
    result = {}
    for winner in winners:
        if winner in result:
            result[winner] += 1
        else:
            result[winner] = 1
    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
            run_performance_comparison(num_games)
        except ValueError:
            print(f"Invalid number of games: {sys.argv[1]}")
            print("Usage: python use_rust_backend.py [num_games]")
    else:
        run_performance_comparison() 