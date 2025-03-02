#!/usr/bin/env python
"""
Test and demonstration of Catanatron Rust integration with Python players.

This file serves both as a test suite and a demonstration of how to use
the Rust implementation of Catanatron with Python players.
"""
import unittest
import time
import random
import os
import argparse
import logging
import sys
import traceback

# Diagnostic information
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current directory:", os.getcwd())
print("\nPython module search paths:")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")

# Set environment variable for Rust logging if not already set
if "RUST_LOG" not in os.environ:
    os.environ["RUST_LOG"] = "info,catanatron_rust=debug"

# Allow command-line arguments to override logging level
def set_rust_log_level(level):
    """Set the RUST_LOG environment variable to control logging level"""
    if level == "debug":
        os.environ["RUST_LOG"] = "debug,catanatron_rust=debug"
    elif level == "info":
        os.environ["RUST_LOG"] = "info,catanatron_rust=info"
    elif level == "warn":
        os.environ["RUST_LOG"] = "warn,catanatron_rust=warn"
    elif level == "error":
        os.environ["RUST_LOG"] = "error,catanatron_rust=error"
    elif level == "quiet":
        os.environ["RUST_LOG"] = "error"
    print(f"Set Rust logging level to: {os.environ['RUST_LOG']}")

# Try to import the Rust implementation with detailed error reporting
RUST_AVAILABLE = False
IMPORT_ERROR_DETAILS = []

try:
    print("Attempting to import catanatron_rust...")
    from catanatron_rust import Game
    print("Successfully imported Game from catanatron_rust")
    
    try:
        print("Attempting to import player classes...")
        # Try relative import first (when running from within the package)
        try:
            from models.player import RandomPlayer, Color, Player
            print("Successfully imported player classes using relative import")
        except ImportError:
            # Try absolute import when installed as a package
            try:
                from catanatron.models.player import RandomPlayer, Color, Player
                print("Successfully imported player classes using absolute import")
            except ImportError as e:
                IMPORT_ERROR_DETAILS.append(f"Failed to import player models: {e}")
                raise
    except ImportError:
        # Define fallback classes if the imports fail
        print("Using fallback player definitions")
        
        # Define a base Color enum
        class Color:
            RED = 0
            BLUE = 1
            WHITE = 2
            ORANGE = 3
            
            def __init__(self, value):
                self.value = value
                self.name = ["RED", "BLUE", "WHITE", "ORANGE"][value]
        
        # Define a base Player class
        class Player:
            def __init__(self, color):
                self.color = color
                self.name = f"Player ({color})"
            
            def decide(self, game_state, playable_actions):
                """Base implementation that must be overridden."""
                raise NotImplementedError("Player subclasses must implement decide()")
        
        # Define a fallback RandomPlayer
        class RandomPlayer(Player):
            def __init__(self, color):
                super().__init__(color)
                self.name = f"RandomPlayer ({color})"
            
            def decide(self, game_state, playable_actions):
                """Simply choose a random action from the available actions."""
                if not playable_actions:
                    return None
                return random.choice(playable_actions)

    try:
        print("Attempting to import Action...")
        # Try relative import first
        try:
            # Try importing from models.enums instead of models.actions
            from models.enums import Action
            print("Successfully imported Action using relative import")
        except ImportError:
            # Try another relative import path
            try:
                # Actions might be in different module
                from models.actions import Action
                print("Successfully imported Action using relative import from actions")
            except ImportError:
                # Try absolute import
                try:
                    from catanatron.models.enums import Action
                    print("Successfully imported Action using absolute import")
                except ImportError as e:
                    IMPORT_ERROR_DETAILS.append(f"Failed to import Action: {e}")
                    print("Using fallback Action definition")
                    # Define a basic Action class
                    class Action:
                        pass
    except ImportError:
        # Basic Action class
        print("Using fallback Action definition")
        class Action:
            pass

    # Simplified Python Game import
    try:
        print("Attempting to import Python Game...")
        # Try relative import first
        try:
            from game import Game as PyGame
            print("Successfully imported Python Game using relative import")
        except ImportError:
            IMPORT_ERROR_DETAILS.append("Failed to import Python Game from relative path")
            print("Python Game import failed, but that's okay for Rust testing")
    except Exception as e:
        IMPORT_ERROR_DETAILS.append(f"Failed to import Python Game: {e}")
        print("Python Game import failed, but that's okay for Rust testing")

    RUST_AVAILABLE = True
except ImportError as e:
    IMPORT_ERROR_DETAILS.append(f"Import error: {e}")
    print(f"Import error: {e}")
    print(traceback.format_exc())
    # Define fallback classes if the imports fail
    
    # Define a base Color enum
    class Color:
        RED = 0
        BLUE = 1
        WHITE = 2
        ORANGE = 3
        
        def __init__(self, value):
            self.value = value
            self.name = ["RED", "BLUE", "WHITE", "ORANGE"][value]
    
    # Define a base Player class
    class Player:
        def __init__(self, color):
            self.color = color
            self.name = f"Player ({color})"
        
        def decide(self, game_state, playable_actions):
            """Base implementation that must be overridden."""
            raise NotImplementedError("Player subclasses must implement decide()")
    
    # Define a fallback RandomPlayer
    class RandomPlayer(Player):
        def __init__(self, color):
            super().__init__(color)
            self.name = f"RandomPlayer ({color})"
        
        def decide(self, game_state, playable_actions):
            """Simply choose a random action from the available actions."""
            if not playable_actions:
                return None
            return random.choice(playable_actions)
    
    # Basic Action class
    class Action:
        pass


def interpret_action_string(action_str):
    """Convert a Rust action string to a readable format for display."""
    # Examples:
    # "Action::BuildSettlement { color: 0, node_id: 12 }"
    # "Action::BuildRoad { color: 1, edge_id: (23, 24) }"
    
    # Basic parsing
    if not action_str or not isinstance(action_str, str):
        return "Unknown action"
    
    # Remove the Action:: prefix
    if action_str.startswith("Action::"):
        action_str = action_str[8:]
    
    # Handle specific action types
    if "BuildSettlement" in action_str:
        # Extract color and node_id
        parts = action_str.split(", ")
        color = parts[0].split(": ")[1].strip()
        node_id = parts[1].split(": ")[1].strip(" }")
        return f"Build Settlement at node {node_id}"
    
    elif "BuildRoad" in action_str:
        # Extract color and edge
        parts = action_str.split(", ")
        color = parts[0].split(": ")[1].strip()
        edge = parts[1].split(": ")[1].strip(" }")
        return f"Build Road along edge {edge}"
    
    elif "BuildCity" in action_str:
        # Extract color and node_id
        parts = action_str.split(", ")
        color = parts[0].split(": ")[1].strip()
        node_id = parts[1].split(": ")[1].strip(" }")
        return f"Build City at node {node_id}"
        
    elif "Roll" in action_str:
        # Check if dice values are specified
        if "None" in action_str:
            return "Roll Dice"
        else:
            # Try to extract dice values
            try:
                dice_part = action_str.split("Some(")[1].split(")")[0]
                return f"Roll Dice: {dice_part}"
            except:
                return "Roll Dice"
    
    # For other actions, just clean up the format a bit
    action_type = action_str.split(" {")[0]
    return action_type


class SimpleAccumulator:
    """Simple accumulator to track game state changes."""
    
    def __init__(self):
        self.actions = []
        self.tick_count = 0
        self.last_state = None
        self.called_before = False
        self.called_after = False
        self.last_winner = None
        
    def before(self, state=None):
        """Called before the game starts."""
        self.called_before = True
        return None
        
    def step(self, state=None, action=None):
        """Called before applying an action."""
        if action is not None:
            self.actions.append(action)
            self.tick_count += 1
        return None
        
    def after(self, state=None):
        """Called after the game ends."""
        self.called_after = True
        self.last_state = state
        return None
        
    def print_action_summary(self, limit=10):
        """Print a summary of the accumulated actions."""
        print(f"\nGame summary: {self.tick_count} total actions")
        
        if not self.actions:
            print("No actions recorded.")
            return
            
        start_idx = max(0, len(self.actions) - limit)
        print(f"Last {min(limit, len(self.actions))} actions:")
        
        for i, action in enumerate(self.actions[start_idx:], start=start_idx + 1):
            action_str = self._format_action(action)
            print(f"#{i}: {action_str}")
            
    def _format_action(self, action):
        """Format an action for printing in a more readable format."""
        try:
            if hasattr(action, "__str__"):
                action_str = str(action)
            else:
                action_str = repr(action)
                
            # Make action strings more readable
            action_str = action_str.replace("Action::", "")
            
            return action_str
        except Exception as e:
            return f"<Unprintable action: {e}>"


class SimplePythonPlayer(Player):
    """A simple Python player that can be passed to Rust."""
    
    def __init__(self, color):
        # Handle different types of color
        if isinstance(color, int):
            self.color = color
            self.color_name = ["RED", "BLUE", "WHITE", "ORANGE"][color]
        else:
            # Assume it's a Color object
            self.color = getattr(color, 'value', color)
            self.color_name = getattr(color, 'name', str(color))
            
        self.action_history = []
        self.name = f"SimplePythonPlayer ({self.color_name})"
        self.log_prefix = f"[Simple {self.color_name}] "
        
    def decide(self, game_state, playable_actions):
        """
        Select a random action from the available ones.
        
        Parameters:
        - game_state: Either a dictionary of state values from Rust or None
        - playable_actions: List of playable actions as strings
        """
        # Print debug info about the game state
        if isinstance(game_state, dict):
            current_color = game_state.get('current_color')
            print(f"{self.log_prefix}Deciding with state info: current_color={current_color}")
            
            # If we received resources, print them
            if 'resources' in game_state:
                resources = game_state['resources']
                print(f"{self.log_prefix}Resources: {resources}")
        else:
            print(f"{self.log_prefix}Deciding without state info")
            
        # Check the format of the actions
        if playable_actions and len(playable_actions) > 0:
            if hasattr(playable_actions[0], "__str__"):
                print(f"{self.log_prefix}Available actions format: {type(playable_actions[0])}")
            else:
                print(f"{self.log_prefix}Available actions format: unknown")
        
        # Log this action
        self.action_history.append(len(playable_actions))
        
        # Choose a random action and return it
        chosen_action = random.choice(playable_actions)
        action_type = str(chosen_action.__class__.__name__ if hasattr(chosen_action, "__class__") else type(chosen_action))
        print(f"{self.log_prefix}Selected {action_type}")
        return chosen_action
    
    def __repr__(self):
        return f"SimplePythonPlayer({self.color_name})"


class SmartPythonPlayer(Player):
    """A slightly smarter Python player that can be used in the Rust game."""
    
    def __init__(self, color):
        # Handle different types of color
        if isinstance(color, int):
            self.color = color
            self.color_name = ["RED", "BLUE", "WHITE", "ORANGE"][color]
        else:
            # Assume it's a Color object
            self.color = getattr(color, 'value', color)
            self.color_name = getattr(color, 'name', str(color))
            
        self.name = f"SmartPythonPlayer ({self.color_name})"
        self.log_prefix = f"[Smart {self.color_name}] "
    
    def decide(self, game_state, playable_actions):
        """Make a more informed decision based on the game state."""
        if not playable_actions:
            return None
            
        print(f"{self.log_prefix}Deciding from {len(playable_actions)} actions")
        
        # Print game state if available
        if isinstance(game_state, dict):
            print(f"{self.log_prefix}Game state: current_color={game_state.get('current_color')}, "
                 f"prompt={game_state.get('action_prompt')}")
            
        # Prioritize winning moves if available
        for action in playable_actions:
            # This would be a real strategy in a complete implementation
            pass
            
        # For now, just choose randomly
        chosen_action = random.choice(playable_actions)
        
        action_type = str(chosen_action.__class__.__name__ if hasattr(chosen_action, "__class__") else type(chosen_action))
        print(f"{self.log_prefix}Selected {action_type}")
        return chosen_action


class ResourcePrioritizingPlayer(Player):
    """A player that prioritizes resource acquisition."""
    
    def __init__(self, color):
        # Handle different types of color
        if isinstance(color, int):
            self.color = color
            self.color_name = ["RED", "BLUE", "WHITE", "ORANGE"][color]
        else:
            # Assume it's a Color object
            self.color = getattr(color, 'value', color)
            self.color_name = getattr(color, 'name', str(color))
            
        self.name = f"ResourcePlayer ({self.color_name})"
        self.log_prefix = f"[Resource {self.color_name}] "
        
    def decide(self, game_state, playable_actions):
        """Make a decision prioritizing resource acquisition."""
        if not playable_actions:
            return None
            
        print(f"{self.log_prefix}Deciding from {len(playable_actions)} actions")
        
        # Print game state if available
        if isinstance(game_state, dict):
            print(f"{self.log_prefix}Game state: current_color={game_state.get('current_color')}, "
                 f"resources={game_state.get('resources')}")
            
        # In a real implementation, this would analyze the board for resource strategies
        # For now, just choose randomly like the other player
        chosen_action = random.choice(playable_actions)
        
        action_type = str(chosen_action.__class__.__name__ if hasattr(chosen_action, "__class__") else type(chosen_action))
        print(f"{self.log_prefix}Selected {action_type}")
        return chosen_action


class StrategicPythonPlayer(Player):
    """A more sophisticated Python player that uses actual strategy."""
    
    def __init__(self, color):
        # Handle different types of color
        if isinstance(color, int):
            self.color = color
            self.color_name = ["RED", "BLUE", "WHITE", "ORANGE"][color]
        else:
            # Assume it's a Color object
            self.color = getattr(color, 'value', color)
            self.color_name = getattr(color, 'name', str(color))
            
        self.name = f"StrategicPlayer ({self.color_name})"
        self.log_prefix = f"[Strategic {self.color_name}] "
        self.game_phase = "early"  # early, mid, late
        self.turn_count = 0
        self.settlement_count = 0
        self.city_count = 0
        self.road_count = 0
        self.owned_resources = {"wood": 0, "brick": 0, "sheep": 0, "wheat": 0, "ore": 0}
        
    def decide(self, game_state, playable_actions):
        """Make a strategic decision based on the game state and available actions."""
        if not playable_actions:
            return None
            
        self.turn_count += 1
        print(f"{self.log_prefix}Deciding from {len(playable_actions)} actions (turn {self.turn_count})")
        
        # Update game knowledge from game state
        if isinstance(game_state, dict):
            # Track resources
            if 'resources' in game_state:
                resources = game_state.get('resources', {})
                self.owned_resources = {
                    "wood": resources.get("wood", 0),
                    "brick": resources.get("brick", 0),
                    "sheep": resources.get("sheep", 0),
                    "wheat": resources.get("wheat", 0),
                    "ore": resources.get("ore", 0),
                }
                
            # Track progress
            if 'victory_points' in game_state:
                vp = game_state.get('victory_points', 0)
                if vp >= 6:
                    self.game_phase = "late"
                elif vp >= 3:
                    self.game_phase = "mid"
                    
            print(f"{self.log_prefix}Game state: phase={self.game_phase}, "
                 f"VP={game_state.get('victory_points', '?')}, "
                 f"resources={self.owned_resources}")
        
        # Group actions by type for better decision making
        action_groups = self._categorize_actions(playable_actions)
        
        # Select action based on strategy for current game phase
        chosen_action = None
        
        # Initial placement strategy
        if "BuildSettlement" in action_groups and self.is_initial_phase(game_state):
            # For initial placements, we want to prioritize spots with good resource diversity
            chosen_action = self._choose_best_settlement(action_groups["BuildSettlement"])
            if chosen_action:
                print(f"{self.log_prefix}Selected initial settlement at strategic location")
                return chosen_action
        
        # Main game strategy
        if self.game_phase == "early":
            chosen_action = self._early_game_strategy(action_groups)
        elif self.game_phase == "mid":
            chosen_action = self._mid_game_strategy(action_groups)
        else:  # late game
            chosen_action = self._late_game_strategy(action_groups)
        
        # If no specific strategy was applied, use a fallback
        if not chosen_action:
            chosen_action = self._default_strategy(action_groups, playable_actions)
        
        action_type = str(chosen_action.__class__.__name__ if hasattr(chosen_action, "__class__") else type(chosen_action))
        print(f"{self.log_prefix}Selected {action_type} through {self.game_phase}-game strategy")
        return chosen_action
    
    def _categorize_actions(self, actions):
        """Group actions by their type for easier strategy decisions."""
        action_groups = {}
        
        for action in actions:
            action_str = str(action)
            
            # Extract the action type (e.g. "BuildSettlement", "BuildRoad", etc.)
            if "::" in action_str:
                action_type = action_str.split("::")[1].split(" ")[0]
            else:
                action_type = "Unknown"
                
            if action_type not in action_groups:
                action_groups[action_type] = []
                
            action_groups[action_type].append(action)
            
        return action_groups
    
    def is_initial_phase(self, game_state):
        """Determine if we're in the initial placement phase."""
        if isinstance(game_state, dict):
            return game_state.get('is_initial_build_phase', False)
        return self.settlement_count < 2 and self.road_count < 2
    
    def _choose_best_settlement(self, settlement_actions):
        """Choose the best settlement location based on strategic value."""
        # In a full implementation, this would analyze the board topology
        # For this demonstration, we'll just pick the first settlement
        if settlement_actions:
            # This would typically analyze node values based on resources, ports, etc.
            return settlement_actions[0]
        return None
    
    def _early_game_strategy(self, action_groups):
        """Focus on resource acquisition and expansion."""
        # Prioritize building settlements for early resource diversity
        if "BuildSettlement" in action_groups and self._has_resources_for("settlement"):
            self.settlement_count += 1
            return action_groups["BuildSettlement"][0]
            
        # Build roads to expand territory
        if "BuildRoad" in action_groups and self._has_resources_for("road"):
            self.road_count += 1
            return action_groups["BuildRoad"][0]
            
        # Roll dice if available
        if "Roll" in action_groups:
            return action_groups["Roll"][0]
            
        # End turn if nothing else to do
        if "EndTurn" in action_groups:
            return action_groups["EndTurn"][0]
            
        return None
    
    def _mid_game_strategy(self, action_groups):
        """Focus on upgrading to cities and getting development cards."""
        # Prioritize building cities
        if "BuildCity" in action_groups and self._has_resources_for("city"):
            self.city_count += 1
            return action_groups["BuildCity"][0]
            
        # Get development cards for advantages
        if "BuyDevelopmentCard" in action_groups and self._has_resources_for("dev_card"):
            return action_groups["BuyDevelopmentCard"][0]
            
        # Build settlements for continued expansion
        if "BuildSettlement" in action_groups and self._has_resources_for("settlement"):
            self.settlement_count += 1
            return action_groups["BuildSettlement"][0]
            
        # Roll dice if available
        if "Roll" in action_groups:
            return action_groups["Roll"][0]
            
        # Trade if needed
        if "MaritimeTrade" in action_groups:
            return self._choose_best_trade(action_groups["MaritimeTrade"])
            
        # End turn if nothing else to do
        if "EndTurn" in action_groups:
            return action_groups["EndTurn"][0]
            
        return None
    
    def _late_game_strategy(self, action_groups):
        """Focus on victory points and blocking opponents."""
        # Prioritize city building for victory points
        if "BuildCity" in action_groups and self._has_resources_for("city"):
            self.city_count += 1
            return action_groups["BuildCity"][0]
            
        # Build settlements for victory points
        if "BuildSettlement" in action_groups and self._has_resources_for("settlement"):
            self.settlement_count += 1
            return action_groups["BuildSettlement"][0]
            
        # Use development cards
        for card_action in ["PlayKnight", "PlayYearOfPlenty", "PlayRoadBuilding", "PlayMonopoly"]:
            if card_action in action_groups:
                return action_groups[card_action][0]
                
        # Roll dice if available
        if "Roll" in action_groups:
            return action_groups["Roll"][0]
            
        # End turn if nothing else to do
        if "EndTurn" in action_groups:
            return action_groups["EndTurn"][0]
            
        return None
    
    def _default_strategy(self, action_groups, all_actions):
        """Fallback strategy when no specific strategy applies."""
        # Try important actions in order of priority
        for action_type in ["Roll", "BuildCity", "BuildSettlement", "BuildRoad", 
                           "BuyDevelopmentCard", "PlayKnight", "EndTurn"]:
            if action_type in action_groups and action_groups[action_type]:
                return action_groups[action_type][0]
                
        # If all else fails, choose randomly
        return random.choice(all_actions)
    
    def _has_resources_for(self, item_type):
        """Check if we have resources for a specific item."""
        if item_type == "road":
            return self.owned_resources["wood"] >= 1 and self.owned_resources["brick"] >= 1
        elif item_type == "settlement":
            return (self.owned_resources["wood"] >= 1 and self.owned_resources["brick"] >= 1 and
                   self.owned_resources["sheep"] >= 1 and self.owned_resources["wheat"] >= 1)
        elif item_type == "city":
            return self.owned_resources["wheat"] >= 2 and self.owned_resources["ore"] >= 3
        elif item_type == "dev_card":
            return (self.owned_resources["sheep"] >= 1 and self.owned_resources["wheat"] >= 1 and
                   self.owned_resources["ore"] >= 1)
        return False
    
    def _choose_best_trade(self, trade_actions):
        """Choose the best maritime trade based on needed resources."""
        # In a full implementation, this would analyze resource needs
        # For this demonstration, we'll just pick the first trade
        if trade_actions:
            return trade_actions[0]
        return None
    
    def __repr__(self):
        return f"StrategicPythonPlayer({self.color_name})"


def run_game_simulation(use_python_players=False):
    """Run a simple game simulation to demonstrate the Rust implementation."""
    if not RUST_AVAILABLE:
        print("Rust implementation not available. Skipping game simulation.")
        for error in IMPORT_ERROR_DETAILS:
            print(f"  - {error}")
        return
        
    try:
        # Track player types for better reporting
        player_types = []
        
        # Check if python players should be used
        if use_python_players:
            print("\nUsing Python player implementation")
        else:
            print("\nUsing Rust RandomPlayer implementation")
        
        # Create players - handle different Color implementations
        players = []
        try:
            # First check if Color is an enum class or a simple class
            is_enum = hasattr(Color, 'RED') and not hasattr(Color, '__init__')
            
            if use_python_players:
                if is_enum:
                    # Simple Color object
                    players = [
                        SmartPythonPlayer(Color.RED),
                        SmartPythonPlayer(Color.BLUE),
                        SmartPythonPlayer(Color.WHITE),
                        SmartPythonPlayer(Color.ORANGE),
                    ]
                    player_types = ["SmartPythonPlayer", "SmartPythonPlayer", "SmartPythonPlayer", "SmartPythonPlayer"]
                else:
                    # Color needs to be instantiated
                    players = [
                        SmartPythonPlayer(Color(0)),  # RED
                        SmartPythonPlayer(Color(1)),  # BLUE
                        SmartPythonPlayer(Color(3)),  # ORANGE
                        SmartPythonPlayer(Color(2)),  # WHITE
                    ]
                    player_types = ["SmartPythonPlayer", "SmartPythonPlayer", "SmartPythonPlayer", "SmartPythonPlayer"]
            else:
                if is_enum:
                    players = [
                        RandomPlayer(Color.RED),
                        RandomPlayer(Color.BLUE),
                        RandomPlayer(Color.WHITE),
                        RandomPlayer(Color.ORANGE),
                    ]
                    player_types = ["RandomPlayer", "RandomPlayer", "RandomPlayer", "RandomPlayer"]
                else:
                    players = [
                        RandomPlayer(Color(0)),  # RED
                        RandomPlayer(Color(1)),  # BLUE
                        RandomPlayer(Color(3)),  # ORANGE
                        RandomPlayer(Color(2)),  # WHITE
                    ]
                    player_types = ["RandomPlayer", "RandomPlayer", "RandomPlayer", "RandomPlayer"]
        except Exception as e:
            print(f"Error creating players: {e}")
            print("Falling back to simpler Color objects...")
            
            # Fallback simple integer Color objects
            if use_python_players:
                players = [
                    SmartPythonPlayer(0),  # RED
                    SmartPythonPlayer(1),  # BLUE
                ]
                player_types = ["SmartPythonPlayer", "SmartPythonPlayer"]
            else:
                players = [
                    RandomPlayer(0),  # RED
                    RandomPlayer(1),  # BLUE
                ]
                player_types = ["RandomPlayer", "RandomPlayer"]
        
        # Print player types summary
        print("\nPlayer setup:")
        for i, player_type in enumerate(player_types):
            print(f"Player {i}: {player_type}")
        
        # Create the accumulator
        accumulator = SimpleAccumulator()
        
        # Create and set up the game
        print(f"\nCreating game with {len(players)} players...")
        game = Game(players)
        print(f"Created game with {game.get_num_players()} players")
        
        # Play the game with the accumulator
        print("\nStarting game simulation...")
        start_time = time.time()
        game.play(accumulators=[accumulator])
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get and display the winner
        winner = game.get_winner()
        
        print("\n" + "=" * 50)
        print(f"Game completed in {elapsed_time:.2f} seconds")
        
        if winner is not None:
            winner_name = player_types[winner] if winner < len(player_types) else f"Player {winner}"
            print(f"WINNER: Player {winner} ({winner_name}) won the game!")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        else:
            print("NO WINNER: Game reached maximum ticks without a winner")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        
        # Show a summary of the last few actions
        accumulator.print_action_summary(limit=15)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during game simulation: {e}")
        print(traceback.format_exc())


def run_mixed_game_simulation():
    """Run a game simulation with both Python and Rust players."""
    if not RUST_AVAILABLE:
        print("Rust implementation not available. Skipping mixed game simulation.")
        for error in IMPORT_ERROR_DETAILS:
            print(f"  - {error}")
        return
        
    try:
        # Track player types for better reporting
        player_types = []
        
        # Create a mix of players - using different player types
        # Handle different Color implementations
        try:
            # First check if Color is an enum class or a simple class
            is_enum = hasattr(Color, 'RED') and not hasattr(Color, '__init__')
            
            if is_enum:
                players = [
                    ResourcePrioritizingPlayer(Color.RED),
                    SmartPythonPlayer(Color.BLUE),
                ]
                player_types = ["ResourcePrioritizingPlayer", "SmartPythonPlayer"]
            else:
                players = [
                    ResourcePrioritizingPlayer(Color(0)),  # RED
                    SmartPythonPlayer(Color(1)),          # BLUE
                ]
                player_types = ["ResourcePrioritizingPlayer", "SmartPythonPlayer"]
        except Exception as e:
            print(f"Error creating players: {e}")
            print("Falling back to simpler Color objects...")
            
            # Fallback simple integer Color objects
            players = [
                ResourcePrioritizingPlayer(0),  # RED
                SmartPythonPlayer(1),          # BLUE
            ]
            player_types = ["ResourcePrioritizingPlayer", "SmartPythonPlayer"]
        
        # Print player types summary
        print("\nPlayer setup:")
        for i, player_type in enumerate(player_types):
            print(f"Player {i}: {player_type}")
        
        # Create the accumulator
        accumulator = SimpleAccumulator()
        
        # Create and set up the game
        print("\nCreating mixed player game...")
        game = Game(players)
        print(f"Created game with {game.get_num_players()} players")
        
        # Play the game with the accumulator
        print("Starting mixed player game simulation...")
        print("(Watch for debugging output from the Python players)")
        start_time = time.time()
        game.play(accumulators=[accumulator])
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get and display the winner
        winner = game.get_winner()
        
        print("\n" + "=" * 50)
        print(f"Mixed player game completed in {elapsed_time:.2f} seconds")
        print(f"Total actions: {len(accumulator.actions)}")
        
        if winner is not None:
            # Display which strategy won
            winner_name = player_types[winner] if winner < len(player_types) else f"Player {winner}"
            print(f"WINNER: Player {winner} ({winner_name}) won the game!")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        else:
            print("NO WINNER: Game reached maximum ticks without a winner")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        
        # Show a summary of the last few actions
        accumulator.print_action_summary(limit=15)    
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during mixed game simulation: {e}")
        print(traceback.format_exc())


def run_value_function_simulation():
    """
    Run a game simulation with ValueFunctionPlayer (from catanatron_experimental) vs Rust RandomPlayer
    """
    if not RUST_AVAILABLE:
        print("Rust implementation not available. Skipping value function simulation.")
        for error in IMPORT_ERROR_DETAILS:
            print(f"  - {error}")
        return
        
    try:
        # Import the ValueFunctionPlayer from catanatron_experimental
        from catanatron_experimental.machine_learning.players.value_function_player import ValueFunctionPlayer
        
        # Track player types for better reporting
        player_types = []
        
        # Set up the game with one ValueFunctionPlayer and one RandomPlayer
        try:
            python_player = ValueFunctionPlayer(Color.RED)
            rust_player = RandomPlayer(Color.BLUE)
            
            players = [python_player, rust_player]
            player_types = ["ValueFunctionPlayer", "RandomPlayer"]
            print(f"Created players: {python_player.name} vs {rust_player.__class__.__name__}")
        except Exception as e:
            print(f"Error creating players: {e}")
            print("Falling back to simpler player initialization...")
            
            # Fallback simple player creation
            python_player = ValueFunctionPlayer(0)  # RED
            rust_player = RandomPlayer(1)  # BLUE
            players = [python_player, rust_player]
            player_types = ["ValueFunctionPlayer", "RandomPlayer"]
        
        # Print player types summary
        print("\nPlayer setup:")
        for i, player_type in enumerate(player_types):
            print(f"Player {i}: {player_type}")
        
        # Create the accumulator
        accumulator = SimpleAccumulator()
        
        # Create and set up the game
        print("\nCreating game with ValueFunctionPlayer vs RandomPlayer...")
        game = Game(players)
        print(f"Created game with {game.get_num_players()} players")
        
        # Play the game with the accumulator
        print("\nStarting game simulation...")
        start_time = time.time()
        game.play(accumulators=[accumulator])
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get and display the winner
        winner = game.get_winner()
        
        print("\n" + "=" * 50)
        print(f"Game completed in {elapsed_time:.2f} seconds")
        
        if winner is not None:
            winner_name = player_types[winner] if winner < len(player_types) else f"Player {winner}"
            print(f"WINNER: Player {winner} ({winner_name}) won the game!")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        else:
            print("NO WINNER: Game reached maximum ticks without a winner")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        
        # Show a summary of the last few actions
        accumulator.print_action_summary(limit=15)
        print("=" * 50)
        
    except ImportError as e:
        print(f"Error importing ValueFunctionPlayer: {e}")
        print("This simulation requires catanatron_experimental package to be installed.")
        print("Make sure you have catanatron_experimental in your Python path.")
        print("Falling back to a standard simulation with Python players.")
        run_game_simulation(use_python_players=True)
    except Exception as e:
        print(f"Error during value function simulation: {e}")
        print(traceback.format_exc())


def run_alpha_beta_simulation():
    """
    Run a game simulation with AlphaBetaPlayer (from catanatron_experimental) vs Rust RandomPlayer
    """
    if not RUST_AVAILABLE:
        print("Rust implementation not available. Skipping alpha beta simulation.")
        for error in IMPORT_ERROR_DETAILS:
            print(f"  - {error}")
        return
        
    try:
        # Import the AlphaBetaPlayer from catanatron_experimental
        from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
        
        # Track player types for better reporting
        player_types = []
        
        # Set up the game with one AlphaBetaPlayer and one RandomPlayer
        try:
            # Create AlphaBetaPlayer with search depth of 2 (as per README)
            python_player = AlphaBetaPlayer(Color.RED, depth=2)
            rust_player = RandomPlayer(Color.BLUE)
            
            players = [python_player, rust_player]
            player_types = ["AlphaBetaPlayer", "RandomPlayer"]
            print(f"Created players: {python_player.name} vs {rust_player.__class__.__name__}")
        except Exception as e:
            print(f"Error creating players: {e}")
            print("Falling back to simpler player initialization...")
            
            # Fallback simple player creation
            python_player = AlphaBetaPlayer(0, depth=2)  # RED
            rust_player = RandomPlayer(1)  # BLUE
            players = [python_player, rust_player]
            player_types = ["AlphaBetaPlayer", "RandomPlayer"]
        
        # Print player types summary
        print("\nPlayer setup:")
        for i, player_type in enumerate(player_types):
            print(f"Player {i}: {player_type}")
        
        # Create the accumulator
        accumulator = SimpleAccumulator()
        
        # Create and set up the game
        print("\nCreating game with AlphaBetaPlayer vs RandomPlayer...")
        game = Game(players)
        print(f"Created game with {game.get_num_players()} players")
        
        # Play the game with the accumulator
        print("\nStarting game simulation...")
        start_time = time.time()
        game.play(accumulators=[accumulator])
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Get and display the winner
        winner = game.get_winner()
        
        print("\n" + "=" * 50)
        print(f"Game completed in {elapsed_time:.2f} seconds")
        
        if winner is not None:
            winner_name = player_types[winner] if winner < len(player_types) else f"Player {winner}"
            print(f"WINNER: Player {winner} ({winner_name}) won the game!")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        else:
            print("NO WINNER: Game reached maximum ticks without a winner")
            print(f"Player types: {', '.join([f'{i}={pt}' for i, pt in enumerate(player_types)])}")
        
        # Show a summary of the last few actions
        accumulator.print_action_summary(limit=15)
        print("=" * 50)
        
    except ImportError as e:
        print(f"Error importing AlphaBetaPlayer: {e}")
        print("This simulation requires catanatron_experimental package to be installed.")
        print("Make sure you have catanatron_experimental in your Python path.")
        print("Falling back to a standard simulation with Python players.")
        run_game_simulation(use_python_players=True)
    except Exception as e:
        print(f"Error during alpha beta simulation: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for the Rust implementation of Catanatron")
    parser.add_argument("-r", "--run", action="store_true", help="Run a simple game simulation")
    parser.add_argument("-m", "--mixed", action="store_true", help="Run a game simulation with mixed player types")
    parser.add_argument("-P", "--python", action="store_true", help="Use Python players instead of Rust players")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-f", "--value", action="store_true", help="Run a game with ValueFunctionPlayer")
    parser.add_argument("-a", "--alpha", action="store_true", help="Run a game with AlphaBetaPlayer (strongest)")
    parser.add_argument("--check-imports", action="store_true", help="Only check imports and report status")
    parser.add_argument("--log-level", choices=["debug", "info", "warn", "error", "quiet"], 
                        default="info", help="Set Rust logging level (use 'quiet' to minimize debug output)")
    args = parser.parse_args()
    
    # Set Rust logging level
    set_rust_log_level(args.log_level)
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    if args.check_imports or not (args.run or args.mixed or args.value or args.alpha):
        # Print import status
        print("\n=== Import Status ===")
        print(f"catanatron_rust: {'Available' if RUST_AVAILABLE else 'Not available'}")
        
        if IMPORT_ERROR_DETAILS:
            print("\nDetailed import errors:")
            for error in IMPORT_ERROR_DETAILS:
                print(f"  - {error}")
            
        print("\n=== Installation Help ===")
        print("If you're having issues with imports, try these steps:")
        print("1. Make sure you've built the Rust library:")
        print("   cd catanatron_rust && maturin develop")
        print("2. Install the Python package in development mode:")
        print("   cd catanatron_core && pip install -e .")
        print("3. Set your PYTHONPATH to include the necessary directories:")
        print("   export PYTHONPATH=$PYTHONPATH:/path/to/catanatron")
        
        if not (args.run or args.mixed or args.value or args.alpha):
            print("\n=== Available Simulations ===")
            print("-r  : Run a simulation with RandomPlayers")
            print("-rP : Run a simulation with SmartPythonPlayers")
            print("-m  : Run a mixed simulation with ResourcePrioritizingPlayer vs SmartPythonPlayer")
            print("-f  : Run a simulation with ValueFunctionPlayer vs RandomPlayer")
            print("-a  : Run a simulation with AlphaBetaPlayer vs RandomPlayer")
            print("\nFor all simulations, player types are clearly tracked and shown in the output.")
            print("Use --log-level=quiet to minimize debug output and focus on game results.")
            sys.exit(0)
    
    if not RUST_AVAILABLE:
        print("\nERROR: The Rust implementation (catanatron_rust) is not available.")
        print("Please build the Rust library and try again.\n")
        sys.exit(1)
    
    # Check for combined flags (-rP instead of -r -P)
    if len(sys.argv) > 1 and any(arg.startswith('-') and len(arg) > 2 and not arg.startswith('--') for arg in sys.argv[1:]):
        print("WARNING: Combined flags like '-rP' are not supported. Please use separate flags like '-r -P' instead.")
    
    # Run game simulations if requested
    if args.alpha:
        run_alpha_beta_simulation()
    elif args.value:
        run_value_function_simulation()
    elif args.run:
        run_game_simulation(use_python_players=args.python)
    elif args.mixed:
        run_mixed_game_simulation()
