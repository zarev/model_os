
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll

# Locals
from api.docker import get_images, get_containers, get_volumes, get_system, inspect_image, inspect_container
from utils.bytes import humanbytes


from terminal.widgets.ItemsList import ItemsList
from terminal.widgets.docker._system import DockerInfo
from terminal.widgets.docker._images import DockerImagesTable
from terminal.widgets.docker._volumes import DockerVolumesTable

class DockerTab(Container):
    """Screen displaying the loaded data through various widgets."""
    CSS_PATH = "../css/main_layout.tcss"

    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.args = args
        self.data = data
        self.loaded = False
        super().__init__(id="tab-docker")

    def compose(self) -> ComposeResult:
        with Horizontal(id="tab-grid"):
            with VerticalScroll(id='tab-main'):
                cnt = Container(id="info-grid", classes="titled-container")
                cnt.border_title = "[b]Info[/]"
                with cnt:
                    yield DockerInfo()
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]Containers[/]"
                with cnt:
                    # TODO: Move into its own component
                    self.data['containers'] = get_containers()
                    yield ItemsList(self.data['containers'], "Containers")
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]Images[/]"
                with cnt:
                    yield DockerImagesTable()
                cnt = Container(classes="titled-container")
                cnt.border_title = "[b]Volumes[/]"
                with cnt:
                    yield DockerVolumesTable()

    def on_mount(self):
        self.title = "OsCli"
        self.sub_title = "Docker"