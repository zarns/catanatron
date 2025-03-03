#!/usr/bin/env python
"""
Enhanced PyO3 Rust Interface Example

This example demonstrates how to use the enhanced PyO3 bindings for Catanatron.
It shows how to create a game, inspect its state, and interact with it using
the Rust backend.
"""
import sys
import os
import random
from typing import List, Dict, Any, Optional

# Try importing from the Rust implementation
try:
    from catanatron_rust import Game, Action, Player as RustPlayer
    RUST_AVAILABLE = True
    print("✅ Rust implementation is available!")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"❌ Rust implementation not available: {e}")
    print("Please build and install the Rust implementation first:")
    print("cd catanatron_rust && maturin develop")
    sys.exit(1)

class SimplePlayer(RustPlayer):
    """A simple player that implements the Rust Player interface"""
    
    def __init__(self, color: int):
        """Initialize the player with a color"""
        super().__init__(color, f"SimplePlayer-{color}")
    
    def decide(self, game_state: Dict[str, Any], playable_actions: List[Action]) -> Action:
        """
        Make a decision based on the game state and available actions.
        
        Args:
            game_state: Dictionary representation of the current game state
            playable_actions: List of actions that can be taken
            
        Returns:
            The chosen action
        """
        print(f"Player {self.color} deciding...")
        print(f"Current phase: {game_state.get('action_prompt', 'Unknown')}")
        
        # Get player resources if available
        if 'players' in game_state:
            color_name = {0: "RED", 1: "BLUE", 2: "ORANGE", 3: "WHITE"}.get(self.color, "UNKNOWN")
            player_data = game_state['players'].get(color_name, {})
            resources = player_data.get('resources', {})
            print(f"Resources: {resources}")
        
        # Choose a random action
        return random.choice(playable_actions)

class SimpleGameObserver:
    """A simple observer that tracks game state and actions"""
    
    def __init__(self):
        """Initialize the observer"""
        self.turns = 0
        self.actions = []
    
    def before(self, state=None):
        """Called before each action"""
        self.turns += 1
        print(f"\n--- Turn {self.turns} ---")
        if state:
            print(f"Current player: {state.get('current_player', 'Unknown')}")
    
    def step(self, state=None, action=None):
        """Called after each action"""
        if action:
            self.actions.append(action)
            print(f"Action taken: {action}")
    
    def after(self, state=None):
        """Called after the game is over"""
        print("\n=== Game Over ===")
        print(f"Total turns: {self.turns}")
        print(f"Total actions: {len(self.actions)}")

def run_demo_game(num_players=4):
    """
    Run a demonstration game using the Rust implementation.
    
    Args:
        num_players: Number of players in the game (2-4)
    """
    print(f"\nStarting a demo game with {num_players} players...\n")
    
    # Create players
    players = [SimplePlayer(i) for i in range(num_players)]
    
    # Create game observer
    observer = SimpleGameObserver()
    
    # Create the game
    game = Game(players, seed=42, discard_limit=7, vps_to_win=10, map_type="BASE")
    
    # Play the game
    winner = game.play([observer])
    
    print(f"\nWinner: Player {winner}")

def inspect_game_state():
    """
    Demonstrate how to inspect and interact with the game state.
    """
    print("\nDemonstrating game state inspection...\n")
    
    # Create players
    players = [SimplePlayer(i) for i in range(4)]
    
    # Create the game
    game = Game(players, seed=42)
    
    # Initialize the game (first tick)
    action = game.play_tick()
    print(f"First action: {action}")
    
    # Get current player
    current_player = game.get_current_player()
    print(f"Current player: {current_player}")
    
    # Get action prompt
    action_prompt = game.get_action_prompt()
    print(f"Action prompt: {action_prompt}")
    
    # Get playable actions
    playable_actions = game.get_playable_actions()
    print(f"Number of playable actions: {len(playable_actions)}")
    if playable_actions:
        print(f"First playable action: {playable_actions[0]}")
    
    # Check if in initial build phase
    is_initial = game.is_initial_build_phase()
    print(f"In initial build phase: {is_initial}")
    
    # Get complete state representation
    state_repr = game.get_state_repr()
    print("\nGame state keys:")
    for key in state_repr.keys():
        print(f"- {key}")
    
    # Get player resources
    player_resources = game.get_resources(0)  # Player 0
    print(f"\nPlayer 0 resources: {player_resources}")

if __name__ == "__main__":
    # Run the demo
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect_game_state()
    else:
        run_demo_game()
        print("\nRun with --inspect to see detailed state inspection") 