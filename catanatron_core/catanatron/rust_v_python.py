#!/usr/bin/env python

import time
import traceback
import sys
import os

# Add parent directories to path to handle imports correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Current directory: {os.getcwd()}")

print("\nPython module search paths:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

try:
    # Import the run_value_function_simulation function from test_rust_game
    from catanatron.test_rust_game import run_value_function_simulation
    print("Successfully imported run_value_function_simulation")
    
    print("\n" + "=" * 50)
    print("Starting ValueFunctionPlayer vs Rust RandomPlayer simulation...")
    print("=" * 50 + "\n")
    
    # Run the simulation
    start_time = time.time()
    winner = run_value_function_simulation()
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 50)
    print(f"Game completed in {elapsed_time:.2f} seconds")
    
    if winner is not None:
        player_types = {0: "ValueFunctionPlayer", 1: "RandomPlayer (Rust)"}
        winner_name = player_types.get(winner, f"Player {winner}")
        print(f"WINNER: Player {winner} ({winner_name}) won the game!")
    else:
        print("NO WINNER: Game reached maximum ticks without a winner")
    print("=" * 50)
    
except Exception as e:
    print(f"Error during simulation: {e}")
    print(traceback.format_exc()) 