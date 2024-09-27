# ---- Monitoring dashboard

## --- Libs
from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Footer, Header, LoadingIndicator, Static
from textual.containers import Container, Horizontal, VerticalScroll

## --- Locals
### -- Api & utils
from api.docker import get_images, get_containers, get_volumes, get_system, inspect_image, inspect_container
from utils.bytes import humanbytes
### -- Docker
from terminal.widgets.docker._system import DockerInfo
from terminal.widgets.docker._images import DockerImages
from terminal.widgets.docker._volumes import DockerVolumes
### -- Host info
from terminal.widgets.info._gpu import GPU
from terminal.widgets.info._cpu import CPU
from terminal.widgets.info._disk import Disk
from terminal.widgets.info._mem import Mem
from terminal.widgets.info._net import Net
from terminal.widgets.info._procs_list import ProcsList


## --- Classes
class HostTab(Container):
    """Screen displaying the loaded data through various widgets."""

    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.args = args
        self.data = data
        super().__init__(id="tab-host")

    def compose(self) -> ComposeResult:
        with Horizontal(id="tab-grid"):
            with VerticalScroll(id='tab-main'):
                cnt = Container(id="info-grid", classes="titled-container")
                cnt.border_title = "[b]Info[/]"
                with cnt:
                    yield DockerInfo()
                yield GPU()
                yield CPU()
                yield Mem()
                yield Net(self.args.net)
                yield Disk()
                # yield DockerImages(self.data['images'])
                # yield DockerVolumes(self.data['volumes'])
            with VerticalScroll(id="tab-pane"):
                yield ProcsList()

    def on_mount(self):
        self.title = "utility_os"
        self.sub_title = "Host information"
        