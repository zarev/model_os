# Libs
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, LoadingIndicator

## --- Locals
### -- Api & utils
from api.docker import get_images, get_containers, get_volumes, get_system

## --- Classes
class LoadingScreen(Screen):
    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.args = args
        self.data = data
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield LoadingIndicator()
        yield Footer()

    async def get_docker_data(self):
        if self.data == {}:
            self.data['images'] = get_images()
            self.data['containers'] = get_containers()
            self.data['volumes'] = get_volumes()
            self.data['system'] = get_system()
        self.log(self.data.keys())
        self.log(self.data)

    async def on_mount(self):
        self.title = "utility_os"
        self.sub_title = "Loading"
        await self.get_docker_data()
        self.app.push_screen('home', [self.args, self.data])