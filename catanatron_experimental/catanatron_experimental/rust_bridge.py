import logging
import sys
from typing import List, Optional, Any, Dict, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Rust implementation
try:
    from catanatron_rust import Game as RustGame
    RUST_AVAILABLE = True
    logger.info("Rust backend available and loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    logger.warning(f"Rust backend not available: {str(e)}")

def is_rust_available() -> bool:
    """Check if the Rust backend is available."""
    return RUST_AVAILABLE

def create_game(players, use_rust=False, **kwargs):
    """
    Factory function to create either a Python or Rust game.
    
    Args:
        players: List of player objects
        use_rust: Whether to use Rust backend if available
        **kwargs: Additional game configuration parameters
    
    Returns:
        A Game instance (either Python or Rust-backed)
    """
    if use_rust and RUST_AVAILABLE:
        logger.info("Creating game with Rust backend")
        return RustGame(players, **kwargs)
    else:
        if use_rust and not RUST_AVAILABLE:
            logger.warning("Rust backend requested but not available. Using Python backend.")
        from catanatron.game import Game
        return Game(players, **kwargs)

class RustAccumulatorAdapter:
    """
    Adapter for Python accumulators to work with Rust backend.
    
    This translates between Rust and Python representations
    of game state and actions for each accumulator method.
    """
    def __init__(self, python_accumulator):
        self.accumulator = python_accumulator
    
    def before(self, state=None):
        """Called before game simulation starts."""
        # Convert state if needed
        return self.accumulator.before(state)
    
    def step(self, state=None, action=None):
        """Called after each game action."""
        # Convert state and action if needed
        return self.accumulator.step(state, action)
    
    def after(self, state=None):
        """Called after game simulation ends."""
        # Convert state if needed
        return self.accumulator.after(state)

def adapt_accumulators_for_backend(accumulators, use_rust):
    """
    Adapt accumulators to work with the specified backend.
    
    Args:
        accumulators: List of accumulator objects
        use_rust: Whether we're using the Rust backend
    
    Returns:
        List of adapted accumulators
    """
    if not accumulators:
        return []
        
    if use_rust and RUST_AVAILABLE:
        return [RustAccumulatorAdapter(acc) for acc in accumulators]
    else:
        return accumulators 