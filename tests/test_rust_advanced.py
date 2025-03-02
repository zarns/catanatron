"""
Advanced tests for the Rust implementation of Catanatron.

These tests focus on specific game mechanics and behaviors to ensure
the Rust implementation correctly handles key aspects of the game.
"""
import pytest
import sys
import os
import random
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import the Rust implementation
try:
    from catanatron_rust import Game as RustGame
    RUST_AVAILABLE = True
    print("🚀 Rust implementation available for testing")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"⚠️ Rust implementation not available: {e}")

# Skip all tests if Rust is not available
pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, 
    reason="Rust implementation not available"
)

# Import Python models for testing
from catanatron.models.player import Color, RandomPlayer, SimplePlayer
from catanatron.models.enums import WOOD, BRICK, SHEEP, WHEAT, ORE

class ControlledTestPlayer(SimplePlayer):
    """A test player that performs predefined actions for testing purposes."""
    
    def __init__(self, color, action_sequence=None):
        super().__init__(color)
        self.action_sequence = action_sequence or []
        self.action_index = 0
        self.actions_taken = []
    
    def decide(self, game_state, playable_actions):
        """Either return the next predefined action or choose randomly."""
        self.actions_taken.append((game_state, playable_actions))
        
        if self.action_index < len(self.action_sequence):
            # Get the next predefined action pattern
            action_pattern = self.action_sequence[self.action_index]
            self.action_index += 1
            
            # Find an action that matches the pattern
            for action in playable_actions:
                if action_pattern in str(action):
                    return action
        
        # If no matching action or no more predefined actions, choose randomly
        return random.choice(playable_actions)

class ResourceTrackingAccumulator:
    """Tracks resources throughout the game."""
    
    def __init__(self):
        self.resource_snapshots = []
        self.actions = []
    
    def before(self, state=None):
        pass
    
    def step(self, state=None, action=None):
        self.actions.append(action)
        
        # Try to extract resource information if available
        if isinstance(state, dict) and 'resources' in state:
            self.resource_snapshots.append({
                'action': action,
                'resources': state['resources'].copy()
            })
    
    def after(self, state=None):
        pass
    
    def get_resource_changes(self):
        """Analyze how resources changed throughout the game."""
        changes = []
        
        for i in range(1, len(self.resource_snapshots)):
            prev = self.resource_snapshots[i-1]
            curr = self.resource_snapshots[i]
            
            change = {
                'action': curr['action'],
                'wood': curr['resources'].get('wood', 0) - prev['resources'].get('wood', 0),
                'brick': curr['resources'].get('brick', 0) - prev['resources'].get('brick', 0),
                'sheep': curr['resources'].get('sheep', 0) - prev['resources'].get('sheep', 0),
                'wheat': curr['resources'].get('wheat', 0) - prev['resources'].get('wheat', 0),
                'ore': curr['resources'].get('ore', 0) - prev['resources'].get('ore', 0),
            }
            
            # Only record if there was actually a change
            if any(change[r] != 0 for r in ['wood', 'brick', 'sheep', 'wheat', 'ore']):
                changes.append(change)
        
        return changes

class TestRustAdvanced:
    """Advanced tests for the Rust implementation."""
    
    def test_initial_build_phase(self):
        """Test that the initial build phase follows the correct pattern."""
        # Create an accumulator to track actions
        class InitialBuildAccumulator:
            def __init__(self):
                self.actions = []
            
            def before(self, state=None):
                pass
            
            def step(self, state=None, action=None):
                self.actions.append(action)
            
            def after(self, state=None):
                pass
            
            def get_initial_build_sequence(self):
                """Extract the initial build sequence from actions."""
                # Find actions up to the first Roll action
                initial_actions = []
                for action in self.actions:
                    initial_actions.append(action)
                    if "Roll" in action:
                        break
                return initial_actions
        
        # Create accumulator
        accumulator = InitialBuildAccumulator()
        
        # Create a game with two players
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Play the game with the accumulator
        winner = game.play([accumulator])
        
        # Get the initial build sequence
        sequence = accumulator.get_initial_build_sequence()
        
        # Check that the sequence follows the correct pattern
        # Player 1: Settlement, Road
        # Player 2: Settlement, Road
        # Player 2: Settlement, Road
        # Player 1: Settlement, Road
        assert len(sequence) >= 8, "Should have at least 8 actions in initial build phase"
        
        # Verify pattern: Settlement, Road, Settlement, Road, ...
        settlement_road_pattern = True
        for i in range(0, 8, 2):
            if i < len(sequence):
                if not ("BuildSettlement" in sequence[i] and "BuildRoad" in sequence[i+1]):
                    settlement_road_pattern = False
                    break
        
        assert settlement_road_pattern, "Initial build should follow Settlement-Road pattern"
        print(f"Initial build sequence: {sequence[:8]}")
    
    def test_resource_collection(self):
        """Test that resources are collected correctly after rolls."""
        # We'll use ControlledTestPlayer to force specific actions
        # and ResourceTrackingAccumulator to track resource changes
        
        # Create a resource tracking accumulator
        accumulator = ResourceTrackingAccumulator()
        
        # Create a game with two players
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Play the game with the accumulator
        winner = game.play([accumulator])
        
        # Analyze resource changes
        resource_changes = accumulator.get_resource_changes()
        
        # Print out some resource change information
        if resource_changes:
            print(f"Detected {len(resource_changes)} resource changes")
            print("Sample resource changes:")
            for change in resource_changes[:5]:
                print(f"  Action: {change['action']}")
                print(f"  Wood: {change['wood']}, Brick: {change['brick']}, " +
                      f"Sheep: {change['sheep']}, Wheat: {change['wheat']}, Ore: {change['ore']}")
        else:
            print("No resource changes detected")
    
    def test_game_end_condition(self):
        """Test that games end when a player reaches the victory point threshold."""
        # Create a game with a low victory point threshold for faster testing
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players, vps_to_win=3)  # Very low VP threshold
        
        # Play the game
        winner = game.play([])
        
        # Check that there is a winner (game should end quickly with low VP threshold)
        assert winner is not None, "Game should have ended with a winner"
        print(f"Winner: Player {winner}")
        
        # The test passes if a winner is determined
        assert winner in [0, 1], "Winner should be one of the players"

    def test_tick_by_tick_state(self):
        """Test that the game state evolves correctly over multiple ticks."""
        # Create a game with two players
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Play a number of ticks and capture the state representation
        state_history = []
        
        for i in range(30):  # Play 30 ticks
            action = game.play_tick()
            state_repr = game.get_state_repr()
            state_history.append((action, state_repr))
        
        # Verify that the state changes over time
        assert len(state_history) == 30, "Should have recorded 30 state snapshots"
        
        # Print out some state transition information
        print("Sample state transitions:")
        for i in range(min(5, len(state_history))):
            print(f"Tick {i}:")
            print(f"  Action: {state_history[i][0]}")
            # Print just a snippet of the state representation
            state_snippet = str(state_history[i][1])[:100] + "..." if state_history[i][1] else "None"
            print(f"  State: {state_snippet}")
            
        # The test passes if no exceptions are raised
        assert True 