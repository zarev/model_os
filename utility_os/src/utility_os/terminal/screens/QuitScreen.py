# Libs
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Label
from textual.containers import Grid

class QuitScreen(Screen):
    """Screen with a dialog to quit."""
    CSS_PATH = "../css/modal_layout.tcss"

    def __init__(self, args: dict) -> None:
        self.args = args
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Are you sure you want to quit?", id="question"),
            Button("Quit", variant="error", id="quit"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "quit":
            self.app.exit()
        else:
            self.app.pop_screen()