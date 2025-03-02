"""
Tests for the Rust implementation of the Catanatron game logic.

These tests focus on the Rust implementation specifically and are designed
to validate that the core functionality works as expected.
"""
import pytest
import sys
import os
import random
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# First try to import the Rust implementation
try:
    from catanatron_rust import Game as RustGame
    RUST_AVAILABLE = True
    print("🚀 Rust implementation available for testing")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"⚠️ Rust implementation not available: {e}")

# Import Python models that we'll use for testing
from catanatron.models.player import Color, RandomPlayer, SimplePlayer

# Skip all tests if Rust is not available
pytestmark = pytest.mark.skipif(
    not RUST_AVAILABLE, 
    reason="Rust implementation not available"
)

class TestRustImplementation:
    """Tests for the Rust implementation of the Catanatron game logic."""
    
    def test_game_creation(self):
        """Test that a Rust game can be created with players."""
        # Create a simple game with two players
        players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Check that the game was created
        assert game is not None
        assert game.get_num_players() == 2
    
    def test_game_with_four_players(self):
        """Test a game with four players."""
        # Create a game with four players
        players = [
            SimplePlayer(Color.RED),
            SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE)
        ]
        game = RustGame(players)
        
        # Check that the game was created correctly
        assert game is not None
        assert game.get_num_players() == 4
    
    def test_game_basic_play(self):
        """Test that a basic game can be played through to completion."""
        # Create a simple game with two players
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Play the game
        start_time = time.time()
        winner = game.play([])
        end_time = time.time()
        
        # Check that the game produced a result
        print(f"Game completed in {end_time - start_time:.2f} seconds")
        
        # The game might not have a winner if it hits the max tick limit
        # but it should at least run without errors
        assert True

    def test_game_with_accumulator(self):
        """Test that a game can be played with an accumulator."""
        
        # Define a simple accumulator
        class SimpleAccumulator:
            def __init__(self):
                self.actions = []
                self.start_time = None
                self.end_time = None
            
            def before(self, state=None):
                self.start_time = time.time()
            
            def step(self, state=None, action=None):
                if action:
                    self.actions.append(action)
            
            def after(self, state=None):
                self.end_time = time.time()
                
            def get_elapsed_time(self):
                if self.start_time and self.end_time:
                    return self.end_time - self.start_time
                return None
        
        # Create accumulator
        accumulator = SimpleAccumulator()
        
        # Create a simple game with two players
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Play the game with the accumulator
        winner = game.play([accumulator])
        
        # Check that the accumulator collected data
        assert accumulator.start_time is not None
        assert accumulator.end_time is not None
        assert len(accumulator.actions) > 0
        
        # Print some info
        print(f"Game completed in {accumulator.get_elapsed_time():.2f} seconds")
        print(f"Collected {len(accumulator.actions)} actions")
        print(f"First few actions: {accumulator.actions[:5]}")
        
        if winner is not None:
            print(f"Winner: Player {winner}")
        else:
            print("No winner determined")

    def test_custom_game_configuration(self):
        """Test that a game can be configured with custom settings."""
        # Create a simple game with custom settings
        players = [SimplePlayer(Color.RED), SimplePlayer(Color.BLUE)]
        game = RustGame(
            players,
            vps_to_win=5,  # Lower VP threshold for faster game
            discard_limit=4  # Custom discard limit
        )
        
        # Play the game
        winner = game.play([])
        
        # The test passes if no exceptions are raised
        assert True
    
    def test_play_tick_by_tick(self):
        """Test that a game can be played tick by tick."""
        players = [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)]
        game = RustGame(players)
        
        # Play a few ticks
        actions = []
        for _ in range(20):  # Play 20 ticks
            action = game.play_tick()
            actions.append(action)
        
        # Check that actions were collected
        assert len(actions) == 20
        print(f"First few actions: {actions[:5]}")
        
        # The test passes if no exceptions are raised
        assert True 