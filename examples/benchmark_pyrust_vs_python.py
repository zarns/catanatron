#!/usr/bin/env python
"""
Benchmark: PyO3 Rust Implementation vs Pure Python Implementation

This script benchmarks the performance difference between the Rust implementation
accessed via PyO3 bindings and the pure Python implementation of Catanatron.
"""
import time
import sys
import os
import random
from typing import List, Dict, Any
import argparse

# Try importing from the Rust implementation
try:
    from catanatron_rust import Game as RustGame, Player as RustPlayer
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    print("❌ Rust implementation not available.")
    print("Please build and install the Rust implementation first:")
    print("cd catanatron_rust && maturin develop")

# Try importing from the Python implementation
try:
    from catanatron.game import Game as PyGame
    from catanatron.models.player import Player as PyPlayer
    from catanatron.models.enums import Color
    PYTHON_AVAILABLE = True
except ImportError:
    PYTHON_AVAILABLE = False
    print("❌ Python implementation not available.")
    print("Please install the Python implementation first:")
    print("pip install -e .")

# Only proceed if at least one implementation is available
if not (RUST_AVAILABLE or PYTHON_AVAILABLE):
    print("Neither implementation is available. Exiting.")
    sys.exit(1)

# Define players for both implementations
class SimpleRustPlayer(RustPlayer):
    """A simple player for the Rust implementation"""
    
    def __init__(self, color):
        super().__init__(color, f"RustPlayer-{color}")
    
    def decide(self, game_state, playable_actions):
        return random.choice(playable_actions)

if PYTHON_AVAILABLE:
    class SimplePyPlayer(PyPlayer):
        """A simple player for the Python implementation"""
        
        def decide(self, game, playable_actions):
            return random.choice(playable_actions)

def benchmark_rust(num_games=10, print_progress=True):
    """
    Benchmark the Rust implementation.
    
    Args:
        num_games: Number of games to play
        print_progress: Whether to print progress
        
    Returns:
        tuple: (total_time, avg_time, games_completed)
    """
    if not RUST_AVAILABLE:
        return None, None, 0
    
    total_time = 0
    games_completed = 0
    
    print(f"Starting Rust benchmark with {num_games} games...")
    start_time_total = time.time()
    
    for i in range(num_games):
        players = [SimpleRustPlayer(j) for j in range(4)]
        game = RustGame(players, seed=i, map_type="BASE")
        
        start_time = time.time()
        winner = game.play([])
        end_time = time.time()
        
        game_time = end_time - start_time
        total_time += game_time
        games_completed += 1
        
        if print_progress and i % max(1, num_games // 10) == 0:
            print(f"Rust Game {i+1}/{num_games} completed in {game_time:.2f}s")
    
    end_time_total = time.time()
    total_elapsed = end_time_total - start_time_total
    avg_time = total_time / games_completed if games_completed > 0 else 0
    
    return total_elapsed, avg_time, games_completed

def benchmark_python(num_games=10, print_progress=True):
    """
    Benchmark the Python implementation.
    
    Args:
        num_games: Number of games to play
        print_progress: Whether to print progress
        
    Returns:
        tuple: (total_time, avg_time, games_completed)
    """
    if not PYTHON_AVAILABLE:
        return None, None, 0
    
    total_time = 0
    games_completed = 0
    
    print(f"Starting Python benchmark with {num_games} games...")
    start_time_total = time.time()
    
    for i in range(num_games):
        players = [SimplePyPlayer(Color(j)) for j in range(4)]
        game = PyGame(players)
        
        start_time = time.time()
        winner = game.play()
        end_time = time.time()
        
        game_time = end_time - start_time
        total_time += game_time
        games_completed += 1
        
        if print_progress and i % max(1, num_games // 10) == 0:
            print(f"Python Game {i+1}/{num_games} completed in {game_time:.2f}s")
    
    end_time_total = time.time()
    total_elapsed = end_time_total - start_time_total
    avg_time = total_time / games_completed if games_completed > 0 else 0
    
    return total_elapsed, avg_time, games_completed

def print_results(rust_results, python_results):
    """
    Print benchmark results.
    
    Args:
        rust_results: Results from Rust benchmark
        python_results: Results from Python benchmark
    """
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    
    if rust_results[0] is not None:
        rust_total, rust_avg, rust_games = rust_results
        print(f"Rust Implementation:")
        print(f"  - Total time: {rust_total:.2f}s")
        print(f"  - Average game time: {rust_avg:.2f}s")
        print(f"  - Games completed: {rust_games}")
    
    if python_results[0] is not None:
        py_total, py_avg, py_games = python_results
        print(f"\nPython Implementation:")
        print(f"  - Total time: {py_total:.2f}s")
        print(f"  - Average game time: {py_avg:.2f}s")
        print(f"  - Games completed: {py_games}")
    
    if rust_results[0] is not None and python_results[0] is not None:
        rust_total, rust_avg, _ = rust_results
        py_total, py_avg, _ = python_results
        
        speedup_total = py_total / rust_total if rust_total > 0 else float('inf')
        speedup_avg = py_avg / rust_avg if rust_avg > 0 else float('inf')
        
        print("\nPerformance Comparison:")
        print(f"  - Rust is {speedup_total:.2f}x faster overall")
        print(f"  - Rust is {speedup_avg:.2f}x faster per game")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Catanatron implementations")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--rust-only", action="store_true", help="Only benchmark Rust implementation")
    parser.add_argument("--python-only", action="store_true", help="Only benchmark Python implementation")
    parser.add_argument("--quiet", action="store_true", help="Don't print progress")
    args = parser.parse_args()
    
    rust_results = (None, None, 0)
    python_results = (None, None, 0)
    
    # Run benchmarks
    if RUST_AVAILABLE and not args.python_only:
        rust_results = benchmark_rust(args.num_games, not args.quiet)
    
    if PYTHON_AVAILABLE and not args.rust_only:
        python_results = benchmark_python(args.num_games, not args.quiet)
    
    # Print results
    print_results(rust_results, python_results)

if __name__ == "__main__":
    main() 