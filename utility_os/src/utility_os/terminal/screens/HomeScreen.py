# ---- Monitoring dashboard

## --- Libs
from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, LoadingIndicator, Static
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Button, ContentSwitcher, Footer, Header, Markdown, TabbedContent, TabPane

## --- Locals
### -- Tabs
from terminal.tabs.DockerTab import DockerTab
from terminal.tabs.HelpTab import HelpTab
from terminal.tabs.HostTab import HostTab
from terminal.tabs.SystemTab import SystemTab
from terminal.tabs.SettingsTab import SettingsTab
from terminal.tabs.UpdatesTab import UpdatesTab
### -- Host info
from terminal.widgets.info._info import InfoLine
## --- Classes
class HomeScreen(Screen):
    """Screen displaying the loaded data through various widgets."""
    CSS_PATH = "../css/home_layout.tcss"
    BINDINGS = [
        ("?", "show_tab('tab-host')", "Host"),
        ("o", "show_tab('tab-system')", "System"),
        ("c", "show_tab('tab-docker')", "Containers"),
        ("m", "show_tab('tab-updates')", "Models"),
        ("?", "show_tab('tab-help')", "Help"),
        ("s", "show_tab('tab-settings')", "Settings"),
    ]

    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.args = args
        self.data = data
        super().__init__()
        

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id='app-grid'):
            with Horizontal(id="top-bar"):  
                with Horizontal(id="tabs-buttons"):  
                    yield Button(":computer: Host", id="tab-host")
                    yield Button(":whale: Docker", id="tab-docker")
                    yield Button(":wrench: System", id="tab-system")
                    yield Button(":download: Updates", id="tab-updates")
                    yield Button(":question: Help", id="tab-help")
                    yield Button(":cog: Settings", id="tab-settings")
                with Horizontal(id="tabs-info"):  
                    yield InfoLine()
            with ContentSwitcher(initial="tab-host"):
                yield HostTab(self.args, self.data)
                yield DockerTab(self.args, self.data)
                yield SystemTab(self.args, self.data)
                yield UpdatesTab(self.args, self.data)
                yield HelpTab(self.args, self.data)
                yield SettingsTab(self.args, self.data)
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.query_one(ContentSwitcher).current = event.button.id

    def action_show_tab(self, tab: str) -> None:
        """Switch to a new tab."""
        self.query_one(ContentSwitcher).current = tab

    def on_mount(self):
        self.title = "utility_os"
        self.sub_title = "Dashboard"
        