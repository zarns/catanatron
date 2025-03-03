#!/usr/bin/env python
"""
State Inspector Example

This script provides an interactive way to explore and visualize
the enhanced state representation in the Catanatron Rust implementation.
"""
import sys
import json
import random
from typing import Dict, Any, List, Optional

try:
    from catanatron_rust import Game, Player, Action
    RUST_AVAILABLE = True
    print("✅ Rust implementation loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"❌ Rust implementation not available: {e}")
    print("Please build and install the Rust implementation first:")
    print("cd catanatron_rust && maturin develop")
    sys.exit(1)

class RandomPlayer(Player):
    """A player that makes random decisions."""
    
    def __init__(self, color: int):
        super().__init__(color, f"RandomPlayer-{color}")
    
    def decide(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> Action:
        return random.choice(playable_actions)

def print_state_section(section_name: str, data: Any):
    """Prints a section of state data with formatting."""
    print(f"\n{'=' * 20} {section_name} {'=' * 20}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and len(value) > 5:
                print(f"{key}: {{{len(value)} items}}")
            elif isinstance(value, list) and len(value) > 5:
                print(f"{key}: [{len(value)} items]")
            else:
                print(f"{key}: {value}")
    elif isinstance(data, list):
        if len(data) > 10:
            print(f"[{len(data)} items]")
            print(f"Sample: {data[:5]} ...")
        else:
            print(data)
    else:
        print(data)

def display_full_state(state: Dict[str, Any]):
    """Displays the full game state in a formatted way."""
    # Basic state information
    basic_info = {
        'current_player': state.get('current_player'),
        'current_player_color': state.get('current_player_color'),
        'is_initial_build_phase': state.get('is_initial_build_phase'),
        'action_prompt': state.get('action_prompt'),
        'has_rolled': state.get('has_rolled')
    }
    print_state_section("BASIC INFO", basic_info)
    
    # Bank information
    if 'bank' in state:
        print_state_section("BANK", state['bank'])
    
    # Robber information
    robber_info = {'robber_tile': state.get('robber_tile')}
    print_state_section("ROBBER", robber_info)
    
    # Player information
    if 'players' in state:
        print_state_section("PLAYERS", state['players'])
        
        # Display detailed information for the current player
        current_color = state.get('current_player_color')
        if current_color and current_color in state['players']:
            current_player = state['players'][current_color]
            print_state_section(f"CURRENT PLAYER ({current_color})", current_player)
    
    # Board state
    board_state = {
        'buildable_nodes': state.get('buildable_nodes', []),
        'buildable_edges': state.get('buildable_edges', [])
    }
    print_state_section("BOARD STATE", board_state)

def step_through_game(num_steps: int = 10):
    """
    Steps through a game one action at a time, displaying the state after each action.
    
    Args:
        num_steps: Maximum number of steps to take
    """
    # Create players
    players = [RandomPlayer(i) for i in range(4)]
    
    # Create game
    game = Game(players, seed=42)
    
    print(f"\nStepping through a game ({num_steps} steps)...")
    
    for i in range(num_steps):
        print(f"\n{'#' * 70}")
        print(f"# STEP {i+1}/{num_steps}")
        print(f"{'#' * 70}")
        
        # Get the current state before the action
        state = game.get_state_repr()
        display_full_state(state)
        
        # Get playable actions
        actions = game.get_playable_actions()
        print(f"\n{'=' * 20} PLAYABLE ACTIONS {'=' * 20}")
        if len(actions) > 5:
            print(f"{len(actions)} actions available")
            print(f"Sample: {actions[:3]} ...")
        else:
            for action in actions:
                print(f"- {action}")
        
        # Play one tick (one action)
        action = game.play_tick()
        print(f"\n{'=' * 20} ACTION TAKEN {'=' * 20}")
        print(action)
        
        # Pause for user input
        input("\nPress Enter to continue to the next step...")
    
    print("\nGame stepping complete!")

def export_state_to_json(filename: str = "game_state.json"):
    """
    Exports a sample game state to a JSON file.
    
    Args:
        filename: Name of the JSON file to create
    """
    # Create players
    players = [RandomPlayer(i) for i in range(4)]
    
    # Create game and play a few ticks to get to an interesting state
    game = Game(players, seed=42)
    
    # Play 10 random ticks
    for _ in range(10):
        game.play_tick()
    
    # Get the current state
    state = game.get_state_repr()
    
    # Write to JSON file
    with open(filename, 'w') as f:
        json.dump(state, f, indent=2)
    
    print(f"\nGame state exported to {filename}")

def main():
    """Main function that parses arguments and runs the appropriate function."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python state_inspector.py step [num_steps]  # Step through a game")
        print("  python state_inspector.py export [filename] # Export state to JSON")
        return
    
    command = sys.argv[1].lower()
    
    if command == "step":
        num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        step_through_game(num_steps)
    elif command == "export":
        filename = sys.argv[2] if len(sys.argv) > 2 else "game_state.json"
        export_state_to_json(filename)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: step, export")

if __name__ == "__main__":
    main() 