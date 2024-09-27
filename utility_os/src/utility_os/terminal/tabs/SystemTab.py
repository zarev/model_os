# ---- Monitoring dashboard

## --- Libs
from rich.text import Text
from time import monotonic
from textual.reactive import reactive
from textual.app import ComposeResult
from textual.widgets import Static, Button, Footer, Header, Static
from textual.containers import Container, VerticalScroll, Horizontal

## --- Classes
class TimeDisplay(Static):
    """A widget to display elapsed time."""

    start_time = reactive(monotonic)
    time = reactive(0.0)
    total = reactive(0.0)

    def on_mount(self) -> None:
        """Event handler called when widget is added to the app."""
        self.update_timer = self.set_interval(1 / 60, self.update_time, pause=True)

    def update_time(self) -> None:
        """Method to update time to current."""
        self.time = self.total + (monotonic() - self.start_time)

    def watch_time(self, time: float) -> None:
        """Called when the time attribute changes."""
        minutes, seconds = divmod(time, 60)
        hours, minutes = divmod(minutes, 60)
        self.update(f"{hours:02,.0f}:{minutes:02.0f}:{seconds:05.2f}")

    def start(self) -> None:
        """Method to start (or resume) time updating."""
        self.start_time = monotonic()
        self.update_timer.resume()

    def stop(self):
        """Method to stop the time display updating."""
        self.update_timer.pause()
        self.total += monotonic() - self.start_time
        self.time = self.total

    def reset(self):
        """Method to reset the time display to zero."""
        self.total = 0
        self.time = 0


class Service(Static):
    """A stopwatch widget."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Event handler called when a button is pressed."""
        button_id = event.button.id
        time_display = self.query_one(TimeDisplay)
        if button_id == "start":
            time_display.start()
            self.add_class("started")
        elif button_id == "stop":
            time_display.stop()
            self.remove_class("started")
        elif button_id == "reset":
            time_display.reset()

    def compose(self) -> ComposeResult:
        """Create child widgets of a stopwatch."""
        yield Button("Start", id="start", variant="success")
        yield Button("Stop", id="stop", variant="error")
        yield Button("Reset", id="reset")
        yield TimeDisplay()

class SystemTab(Container):
    """Screen with a Loader while data is being fetched."""
    CSS_PATH = "../css/main_layout.tcss"

    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.data = data
        self.args = args
        super().__init__(id="tab-system")

    def compose(self) -> ComposeResult:
        with Horizontal(id="tab-grid"):
            with VerticalScroll(id='tab-main'):
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc1[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc2[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc3[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc4[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc5[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc6[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc7[/]"
                with cnt:
                    yield Static("Test")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]svc8[/]"
                with cnt:
                    yield Static("Test")
            with VerticalScroll(id="tab-pane"):
               yield Service()
               yield Service()
               yield Service()

    def on_mount(self):
        self.title = "OsCli"
        self.sub_title = "System tab"
        