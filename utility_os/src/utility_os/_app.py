# ---- Main app

## --- Libs
import argparse
from sys import version_info
from textual import on, events
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Placeholder, OptionList, ContentSwitcher, Footer, Header, Markdown, TabbedContent
from textual.containers import Horizontal, Vertical
from textual.widgets.option_list import Option, Separator

## --- Locals
### -- Api & utils
from __about__ import __version__
### -- Screens
from terminal.screens.HomeScreen import HomeScreen
from terminal.screens.QuitScreen import QuitScreen
from terminal.screens.LoadingScreen import LoadingScreen

## --- Classes
class UtilityOs(App):
    """`utility_os` allows you to run and manage `model_os` effortlessly."""

    # CSS_PATH = "terminal/css/main_layout.tcss"

    BINDINGS = [
        ("t", "toggle_dark", "Toggle dark mode"),
        ("h", "push_screen('home')", "Home"),
        ("q", "push_screen('quit')", "Quit (clean exit)"),
        ("escape", "quit", "Quit (force)"),
    ]

    def __init__(self, args: dict) -> None:
        self.args = args
        super().__init__()

    def on_mount(self) -> None:
        self.title = "utility_os"
        self.sub_title = "Manage your model_os instances easily"
        # Install all screen instances
        self.install_screen(QuitScreen(self.args), name="quit")
        self.install_screen(HomeScreen(self.args), name="home")
        self.install_screen(LoadingScreen(self.args), name="load")
        # Navigate to Home screen
        self.push_screen('load')

def run(argv=None):
    parser = argparse.ArgumentParser(
        description="`utility_os` allows you to run and manage `model_os` effortlessly.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=_get_version_text(),
        help="display version information",
    )

    parser.add_argument(
        "--net",
        "-n",
        type=str,
        default=None,
        help="network interface to display (default: auto)",
    )

    utility_instance = UtilityOs(parser.parse_args(argv))
    utility_instance.run()

def _get_version_text():
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    return "\n".join(
        [
            f"utility_os {__version__} [Python {python_version}]",
            "Copyright (c) 2024 hexifox",
        ]
    )

if "__main__":
    run()