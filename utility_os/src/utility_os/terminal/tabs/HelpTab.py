# Libs
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Placeholder, Footer, Header
from textual.containers import Container

class HelpTab(Container):
    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.args = args
        self.data = data
        super().__init__(id="tab-help")

    def compose(self) -> ComposeResult:
        yield Placeholder("Help Tab")