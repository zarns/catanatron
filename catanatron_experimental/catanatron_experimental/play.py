import os
import sys
import importlib.util
from dataclasses import dataclass
from typing import Literal, Union

# Add all necessary paths to Python path to ensure modules can be found
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_experimental'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_rust'))

# Debug print to verify paths
print("Python path:")
for p in sys.path:
    print(f"  {p}")
print(f"BASE_DIR: {BASE_DIR}")
print(f"Current directory: {os.getcwd()}")
print(f"__file__: {__file__}")

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.progress import Progress, BarColumn, TimeRemainingColumn
from rich import box
from rich.console import Console
from rich.theme import Theme
from rich.text import Text

# Try importing required modules with better error handling
try:
    from catanatron.game import Game, Color
    from catanatron.models.player import Player, RandomPlayer
    from catanatron.models.map_instance import build_map
    from catanatron.state_functions import get_actual_victory_points
except ImportError:
    # Use custom imports if catanatron isn't installed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "catanatron_core"))
    from catanatron.game import Game, Color
    from catanatron.models.player import Player, RandomPlayer
    from catanatron.models.map_instance import build_map
    from catanatron.state_functions import get_actual_victory_points

# Import our Rust bridge module
try:
    from catanatron_experimental.rust_bridge import (
        create_game,
        adapt_accumulators_for_backend,
        is_rust_available,
    )
except ImportError:
    print("Error: Could not import rust_bridge. Creating fallback functions.")
    # Create fallback functions if rust_bridge is not available
    def is_rust_available():
        return False
    
    def create_game(players, use_rust=False, **kwargs):
        from catanatron.game import Game
        return Game(players, **kwargs)
    
    def adapt_accumulators_for_backend(accumulators, use_rust):
        return accumulators

# try to suppress TF output before any potentially tf-importing modules
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from catanatron_experimental.utils import ensure_dir, formatSecs
from catanatron_experimental.cli.cli_players import (
    CUSTOM_ACCUMULATORS,
    player_help_table,
    CLI_PLAYERS,
    parse_player_arg,
    get_player,
    RUST_AVAILABLE,
    RustRandomPlayerProxy as RustRandomPlayer,
)
from catanatron_experimental.cli.accumulators import (
    JsonDataAccumulator,
    CsvDataAccumulator,
    DatabaseAccumulator,
    StatisticsAccumulator,
    VpDistributionAccumulator,
)
from catanatron_experimental.cli.simulation_accumulator import SimulationAccumulator


custom_theme = Theme(
    {
        "progress.remaining": "",
        "progress.percentage": "",
        "bar.complete": "green",
        "bar.finished": "green",
    }
)
console = Console(theme=custom_theme)


class CustomTimeRemainingColumn(TimeRemainingColumn):
    """Renders estimated time remaining according to show_time field."""

    def render(self, task):
        """Show time remaining."""
        show = task.fields.get("show_time", True)
        if not show:
            return Text("")
        return super().render(task)


@click.command()
@click.option("-n", "--num", default=5, help="Number of games to play.")
@click.option(
    "--players",
    default="R,R,R,R",
    help="""
    Comma-separated players to use. Use ':' to set player-specific params.
    (e.g. --players=R,G:25,AB:2:C,W).\n
    See player legend with '--help-players'.
    """,
)
@click.option(
    "--code",
    default=None,
    help="Path to file with custom Players and Accumulators to import and use.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    help="Directory where to save game data.",
)
@click.option(
    "--json",
    default=None,
    is_flag=True,
    help="Save game data in JSON format.",
)
@click.option(
    "--csv", default=False, is_flag=True, help="Save game data in CSV format."
)
@click.option(
    "--db",
    default=False,
    is_flag=True,
    help="""
        Save game in PGSQL database.
        Expects docker-compose provided database to be up and running.
        This allows games to be watched.
        """,
)
@click.option(
    "--rust",
    default=False,
    is_flag=True,
    help="Use the Rust backend for faster simulation (if available)",
)
@click.option(
    "--config-discard-limit",
    default=7,
    help="Sets Discard Limit to use in games.",
)
@click.option(
    "--config-vps-to-win",
    default=10,
    help="Sets Victory Points needed to win games.",
)
@click.option(
    "--config-map",
    default="BASE",
    type=click.Choice(["BASE", "MINI", "TOURNAMENT"], case_sensitive=False),
    help="Sets Map to use. MINI is a 7-tile smaller version. TOURNAMENT uses a fixed balanced map.",
)
@click.option(
    "--quiet",
    default=False,
    is_flag=True,
    help="Silence console output. Useful for debugging.",
)
@click.option(
    "--help-players",
    default=False,
    type=bool,
    help="Show player codes and exits.",
    is_flag=True,
)
@click.option(
    "--benchmark",
    default=False,
    is_flag=True,
    help="Run both Python and Rust backends to compare performance",
)
def simulate(
    num,
    players,
    code,
    output,
    json,
    csv,
    db,
    rust,
    config_discard_limit,
    config_vps_to_win,
    config_map,
    quiet,
    help_players,
    benchmark,
):
    """
    Catan Bot Simulator.
    Catanatron allows you to simulate millions of games at scale
    and test bot strategies against each other.

    Examples:\n\n
        catanatron-play --players=R,R,R,R --num=1000\n
        catanatron-play --players=W,W,R,R --num=50000 --output=data/ --csv\n
        catanatron-play --players=VP,F --num=10 --output=data/ --json\n
        catanatron-play --players=W,F,AB:3 --num=1 --csv --json --db --quiet\n
        catanatron-play --players=R,R,R,R --num=100 --rust
    """
    if code:
        abspath = os.path.abspath(code)
        spec = importlib.util.spec_from_file_location("module.name", abspath)
        if spec is not None and spec.loader is not None:
            user_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_module)

    if help_players:
        return Console().print(player_help_table())
    if output and not (json or csv):
        return print("--output requires either --json or --csv to be set")

    console = Console()
    
    # Check if the Rust backend is available
    rust_backend_available = is_rust_available()
    if rust:
        if rust_backend_available:
            console.print("[bold green]Using Rust backend for game simulation! 🚀[/bold green]")
        else:
            console.print("[bold red]Rust backend requested but not available. Falling back to Python.[/bold red]")
            
    # Parse player codes from the comma-separated string
    player_configs = parse_player_arg(players)
    
    # Get the color list
    player_color_names = ["RED", "BLUE", "ORANGE", "WHITE"]
    colors = (
        [Color[color_name] for color_name in player_color_names[: len(player_configs)]]
        if len(player_configs) <= 4
        else []
    )

    # Create the players
    players = []
    for i, (player_code, params) in enumerate(player_configs):
        color = colors[i] if i < len(colors) else None
        
        # Special case for RR to provide better error handling
        if player_code == "RR" and rust:
            try:
                from catanatron_experimental.cli.cli_players import RustRandomPlayerProxy
                if color is not None:
                    player = RustRandomPlayerProxy(color, f"RustRandom {i}")
                    print(f"Created RustRandomPlayer with color={color}")
                    players.append(player)
                    continue
            except Exception as e:
                print(f"Error handling RustRandomPlayer: {e}, trying normal creation")
                
        # Normal player creation for all other players
        cli_player = get_player(player_code)
        if color is not None:
            params = [color] + list(params)
        try:
            player = cli_player.import_fn(*params)
            players.append(player)
        except Exception as e:
            print(f"Error creating player with {player_code}: {e}")
            return

    # Optimize players for Rust if the Rust backend is being used
    if rust and rust_backend_available:
        # First mark all players with _rust_color for easy access
        prepare_players_for_rust(players)
        
        # Then try to create Rust native players where possible
        players = create_rust_native_players(players, rust_backend_available)

    output_options = OutputOptions(output, csv, json, db)
    game_config = GameConfigOptions(config_discard_limit, config_vps_to_win, config_map)
    
    # Run benchmark mode if requested
    if benchmark and is_rust_available():
        console.print("[bold]Running benchmark: Python vs Rust[/bold]")
        
        # Python run
        console.print("\n[bold yellow]Running with Python backend...[/bold yellow]")
        import time
        py_start_time = time.time()
        py_result = play_batch(
            num,
            players,
            output_options,
            game_config,
            quiet,
            use_rust=False,
        )
        py_duration = time.time() - py_start_time
        py_games_per_sec = num / py_duration
        
        # Rust run
        console.print("\n[bold blue]Running with Rust backend...[/bold blue]")
        rust_start_time = time.time()
        rust_result = play_batch(
            num,
            players,
            output_options,
            game_config,
            quiet,
            use_rust=True,
        )
        rust_duration = time.time() - rust_start_time
        rust_games_per_sec = num / rust_duration
        
        # Print comparison
        console.print("\n[bold]Benchmark Results:[/bold]")
        console.print(f"Python: {py_duration:.2f} seconds ({py_games_per_sec:.2f} games/sec)")
        console.print(f"Rust:   {rust_duration:.2f} seconds ({rust_games_per_sec:.2f} games/sec)")
        console.print(f"Speedup: {py_duration/rust_duration:.2f}x faster with Rust")
        
        return py_result and rust_result
    else:
        # Regular run (either Python or Rust)
        return play_batch(
            num,
            players,
            output_options,
            game_config,
            quiet,
            use_rust=rust and rust_backend_available,
        )


@dataclass(frozen=True)
class OutputOptions:
    """Class to keep track of output CLI flags"""

    output: Union[str, None] = None  # path to store files
    csv: bool = False
    json: bool = False
    db: bool = False


@dataclass(frozen=True)
class GameConfigOptions:
    discard_limit: int = 7
    vps_to_win: int = 10
    map_instance: Literal["BASE", "TOURNAMENT", "MINI"] = "BASE"


COLOR_TO_RICH_STYLE = {
    Color.RED: "red",
    Color.BLUE: "blue",
    Color.ORANGE: "yellow",
    Color.WHITE: "white",
}


def rich_player_name(player):
    style = COLOR_TO_RICH_STYLE[player.color]
    return f"[{style}]{player}[/{style}]"


def rich_color(color):
    if color is None:
        return ""
    style = COLOR_TO_RICH_STYLE[color]
    return f"[{style}]{color.value}[/{style}]"


def prepare_players_for_rust(players):
    """Prepare players for use with the Rust backend by adding _rust_color attribute."""
    # Map player colors to numeric values (RED=0, BLUE=1, ORANGE=2, WHITE=3)
    color_map = {
        "RED": 0,
        "BLUE": 1,
        "ORANGE": 2,
        "WHITE": 3,
    }
    
    for player in players:
        if hasattr(player, '_rust_color'):
            print(f"Player {player.name} already has _rust_color = {player._rust_color}")
            continue
        
        if hasattr(player, 'color'):
            color_name = str(player.color.name) if hasattr(player.color, 'name') else str(player.color)
            if color_name in color_map:
                player._rust_color = color_map[color_name]
                print(f"Set _rust_color = {player._rust_color} for player {player.name}")
            else:
                print(f"Warning: Unknown color {color_name} for player {player.name}")
    
    return players


def create_rust_native_players(python_players, rust_available):
    """
    Replace RandomPlayers with RustRandomPlayer instances if Rust is available.
    
    Args:
        python_players: List of Python player objects
        rust_available: Boolean indicating if Rust backend is available
        
    Returns:
        List of players with RandomPlayers optionally replaced by RustRandomPlayer
    """
    # If Rust is not available, just return the original players
    if not rust_available:
        print("Rust backend not available, using Python players")
        return python_players
    
    # We already have RustRandomPlayerProxy instances marked with is_rust_player=True attribute
    # Simply return the existing players as they're already optimized
    
    # Check if any players need to be upgraded to Rust
    any_standard_players = False
    for player in python_players:
        if hasattr(player, 'is_rust_player') and player.is_rust_player:
            # These players are already configured for Rust
            print(f"Player {player} is already configured for Rust")
        elif isinstance(player, RandomPlayer) and not hasattr(player, 'is_rust_player'):
            # This is a regular RandomPlayer that could be upgraded
            any_standard_players = True
    
    if not any_standard_players:
        # If no standard players to upgrade, return as is
        print("All applicable players already optimized for Rust")
        return python_players
    
    # Otherwise create new list with potentially replaced players  
    try:
        # Import RustRandomPlayerProxy
        from catanatron_experimental.cli.cli_players import RustRandomPlayerProxy
        
        rust_players = []
        for player in python_players:
            # If this player is already a Rust-optimized player, keep it
            if hasattr(player, 'is_rust_player') and player.is_rust_player:
                rust_players.append(player)
            # If this is a RandomPlayer, replace it with a RustRandomPlayer
            elif isinstance(player, RandomPlayer):
                # Create a RustRandomPlayer with the same color
                print(f"Creating RustRandomPlayer to replace {player}")
                rust_player = RustRandomPlayerProxy(player.color, name=player.name)
                rust_players.append(rust_player)
            else:
                # Keep the original player
                rust_players.append(player)
                
        return rust_players
    except Exception as e:
        print(f"Error creating Rust players: {e}")
        return python_players


def play_batch_core(num_games, players, game_config, accumulators=[], use_rust=False):
    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.before_all()

    # Prepare players for Rust if needed
    if use_rust:
        players = prepare_players_for_rust(players)
        
        # Try to create native Rust players if available
        rust_available = is_rust_available()
        if rust_available:
            print("Attempting to create native Rust players...")
            native_rust_players = create_rust_native_players(players, rust_available)
            if native_rust_players != players:
                print(f"Successfully created {len(native_rust_players)} native Rust players")
                players = native_rust_players

    for _ in range(num_games):
        for player in players:
            if hasattr(player, 'reset_state'):
                player.reset_state()
        map_instance = build_map(game_config.map_instance)
        
        # Use the Rust backend if requested and available
        game = create_game(
            players,
            use_rust=use_rust,
            discard_limit=game_config.discard_limit,
            vps_to_win=game_config.vps_to_win,
            map_instance=map_instance,
        )
        
        # Adapt accumulators for the backend
        adapted_accumulators = adapt_accumulators_for_backend(accumulators, use_rust)
        
        game.play(adapted_accumulators)
        yield game

    for accumulator in accumulators:
        if isinstance(accumulator, SimulationAccumulator):
            accumulator.after_all()


def play_batch(
    num_games,
    players,
    output_options=None,
    game_config=None,
    quiet=False,
    use_rust=False,
):
    output_options = output_options or OutputOptions()
    game_config = game_config or GameConfigOptions()

    statistics_accumulator = StatisticsAccumulator()
    vp_accumulator = VpDistributionAccumulator()
    accumulators = [statistics_accumulator, vp_accumulator]
    if output_options.output:
        ensure_dir(output_options.output)
    if output_options.output and output_options.csv:
        accumulators.append(CsvDataAccumulator(output_options.output))
    if output_options.output and output_options.json:
        accumulators.append(JsonDataAccumulator(output_options.output))
    if output_options.db:
        accumulators.append(DatabaseAccumulator())
    for accumulator_class in CUSTOM_ACCUMULATORS:
        accumulators.append(accumulator_class(players=players, game_config=game_config))

    console = Console(theme=custom_theme)
    
    # Setup timing measurement
    import time
    start_time = time.time()
    
    if quiet:
        for _ in play_batch_core(num_games, players, game_config, accumulators, use_rust):
            pass
    else:
        # Setup progress tracking
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            CustomTimeRemainingColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task(f"[cyan]Running {num_games} games...", total=num_games)
            player_tasks = [
                progress.add_task(rich_player_name(player), total=num_games, show_time=False)
                for player in players
            ]
            
            for i, game in enumerate(play_batch_core(num_games, players, game_config, accumulators, use_rust)):
                progress.update(main_task, advance=1)
                winner = game.winning_color()
                if winner is not None:
                    for j, player in enumerate(players):
                        if player.color.value == winner:
                            progress.update(player_tasks[j], advance=1)

    player_statistics = statistics_accumulator.player_statistics
    
    # Calculate and display timing information
    elapsed_time = time.time() - start_time
    games_per_second = num_games / elapsed_time
    
    if not quiet:
        console.print(f"\nCompleted {num_games} games in {formatSecs(elapsed_time)}")
        console.print(f"Performance: {games_per_second:.2f} games/second")
        if use_rust:
            console.print(f"[green]Using Rust backend[/green]")
        else:
            console.print(f"[yellow]Using Python backend[/yellow]")
    
    # ===== Game Details
    last_n = 10
    actual_last_n = min(last_n, num_games)
    table = Table(title=f"Last {actual_last_n} Games", box=box.MINIMAL)
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("SEATING")
    table.add_column("TURNS", justify="right")
    for player in players:
        table.add_column(f"{player.color.value} VP", justify="right")
    table.add_column("WINNER")
    if output_options.db:
        table.add_column("LINK", overflow="fold")

    # Display last N games
    for i, game in enumerate(statistics_accumulator.games[-actual_last_n:]):
        # Adapt access based on whether it's a Python or Rust game
        if hasattr(game, 'state'):
            # Python game
            seating = "".join([c.value[0] for c in game.state.colors])
            turns = game.state.num_turns
            winner = game.winning_color()
        else:
            # Rust game
            seating = "".join([str(i) for i in range(len(players))])  # placeholder
            turns = 0  # placeholder
            winner = game.get_winner()
        
        row = [
            str(num_games - (actual_last_n - i) + 1),
            seating,
            str(turns),
        ]
        
        for player in players:
            if hasattr(game, 'state'):
                # Python game
                points = get_actual_victory_points(game.state, player.color)
            else:
                # Rust game - get points directly from player stats
                points = statistics_accumulator.results_by_player[player.color][-actual_last_n + i]
            row.append(str(points))
            
        if winner is not None:
            row.append(rich_color(winner))
        else:
            row.append("None")
            
        if output_options.db:
            row.append(accumulators[-1].link)
            
        table.add_row(*row)

    console.print(table)

    # ===== PLAYER SUMMARY
    table = Table(title="Player Summary", box=box.MINIMAL)
    table.add_column("", no_wrap=True)
    table.add_column("WINS", justify="right")
    table.add_column("AVG VP", justify="right")
    table.add_column("AVG SETTLES", justify="right")
    table.add_column("AVG CITIES", justify="right")
    table.add_column("AVG ROAD", justify="right")
    table.add_column("AVG ARMY", justify="right")
    table.add_column("AVG DEV VP", justify="right")
    for player in players:
        vps = statistics_accumulator.results_by_player[player.color]
        avg_vps = sum(vps) / len(vps)
        avg_settlements = vp_accumulator.get_avg_settlements(player.color)
        avg_cities = vp_accumulator.get_avg_cities(player.color)
        avg_largest = vp_accumulator.get_avg_largest(player.color)
        avg_longest = vp_accumulator.get_avg_longest(player.color)
        avg_devvps = vp_accumulator.get_avg_devvps(player.color)
        table.add_row(
            rich_player_name(player),
            str(statistics_accumulator.wins[player.color]),
            f"{avg_vps:.2f}",
            f"{avg_settlements:.2f}",
            f"{avg_cities:.2f}",
            f"{avg_longest:.2f}",
            f"{avg_largest:.2f}",
            f"{avg_devvps:.2f}",
        )
    console.print(table)

    # ===== GAME SUMMARY
    avg_ticks = f"{statistics_accumulator.get_avg_ticks():.2f}"
    avg_turns = f"{statistics_accumulator.get_avg_turns():.2f}"
    avg_duration = formatSecs(statistics_accumulator.get_avg_duration())
    table = Table(box=box.MINIMAL, title="Game Summary")
    table.add_column("AVG TICKS", justify="right")
    table.add_column("AVG TURNS", justify="right")
    table.add_column("AVG DURATION", justify="right")
    table.add_row(avg_ticks, avg_turns, avg_duration)
    console.print(table)

    if output_options.output and output_options.csv:
        console.print(f"GZIP CSVs saved at: [green]{output_options.output}[/green]")

    return (
        dict(statistics_accumulator.wins),
        dict(statistics_accumulator.results_by_player),
        statistics_accumulator.games,
    )


if __name__ == "__main__":
    simulate()
