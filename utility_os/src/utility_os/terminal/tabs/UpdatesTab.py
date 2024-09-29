# ---- Monitoring dashboard

## --- Libs
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Placeholder, TabPane
from textual.containers import Container

## --- Classes
class UpdatesTab(Container):
    """Screen with a Loader while data is being fetched."""
    CSS_PATH = "../css/main_layout.tcss"

    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.data = data
        self.args = args
        super().__init__(id="tab-updates")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Placeholder('Updates')

    def on_mount(self):
        self.title = "OsCli"
        self.sub_title = "Updates tab"
        