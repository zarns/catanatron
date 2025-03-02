import time
import argparse
import statistics
import logging
from typing import List, Dict, Any
import sys

from catanatron.models.player import RandomPlayer, Color
from catanatron_experimental.rust_bridge import create_game, is_rust_available

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark(num_games=10, num_players=4, verbose=True):
    """
    Run a benchmark comparing Python and Rust backends.
    
    Args:
        num_games: Number of games to run for each backend
        num_players: Number of players in each game
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "python": {"times": [], "winners": []},
        "rust": {"times": [], "winners": []}
    }
    
    # Create players
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:num_players]
    players = [RandomPlayer(color) for color in colors]
    
    # Benchmark Python backend
    if verbose:
        print(f"\nRunning {num_games} games with Python backend...")
    
    start_total = time.time()
    for i in range(num_games):
        start_time = time.time()
        game = create_game(players, use_rust=False)
        winner = game.play()
        end_time = time.time()
        
        game_time = end_time - start_time
        results["python"]["times"].append(game_time)
        results["python"]["winners"].append(winner)
        
        if verbose:
            print(f"  Game {i+1}: {game_time:.4f} seconds, winner: {winner}")
    
    python_total = time.time() - start_total
    
    # Calculate Python stats
    python_avg = statistics.mean(results["python"]["times"])
    python_std = statistics.stdev(results["python"]["times"]) if len(results["python"]["times"]) > 1 else 0
    
    if verbose:
        print(f"Python total time: {python_total:.2f} seconds")
        print(f"Python average: {python_avg:.4f} seconds (±{python_std:.4f})")
    
    # Benchmark Rust backend if available
    if is_rust_available():
        if verbose:
            print(f"\nRunning {num_games} games with Rust backend...")
        
        start_total = time.time()
        for i in range(num_games):
            start_time = time.time()
            game = create_game(players, use_rust=True)
            winner = game.play()
            end_time = time.time()
            
            game_time = end_time - start_time
            results["rust"]["times"].append(game_time)
            results["rust"]["winners"].append(winner)
            
            if verbose:
                print(f"  Game {i+1}: {game_time:.4f} seconds, winner: {winner}")
        
        rust_total = time.time() - start_total
        
        # Calculate Rust stats
        rust_avg = statistics.mean(results["rust"]["times"])
        rust_std = statistics.stdev(results["rust"]["times"]) if len(results["rust"]["times"]) > 1 else 0
        
        if verbose:
            print(f"Rust total time: {rust_total:.2f} seconds")
            print(f"Rust average: {rust_avg:.4f} seconds (±{rust_std:.4f})")
            
        # Calculate speedup
        speedup = python_avg / rust_avg
        speedup_total = python_total / rust_total
        if verbose:
            print(f"\nAverage game speedup: {speedup:.2f}x")
            print(f"Total benchmark speedup: {speedup_total:.2f}x")
        
        results["speedup"] = speedup
        results["speedup_total"] = speedup_total
    else:
        if verbose:
            print("Rust backend not available for benchmarking")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Catanatron backends")
    parser.add_argument("--games", type=int, default=10, help="Number of games to run")
    parser.add_argument("--players", type=int, default=4, help="Number of players per game")
    parser.add_argument("--silent", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()
    
    print("\n=============== CATANATRON BENCHMARK ===============")
    print(f"Running benchmark with {args.games} games and {args.players} players")
    print("===================================================\n")
    
    run_benchmark(num_games=args.games, num_players=args.players, verbose=not args.silent)
    
    print("\n===================================================")
    print("Benchmark complete")
    print("===================================================\n")

if __name__ == "__main__":
    main() 