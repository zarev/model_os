from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static, DataTable, LoadingIndicator
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from typing import Union

# Locals
from api.docker import get_images, get_containers, get_volumes, get_system, inspect_image, inspect_container
from utils.bytes import humanbytes


from terminal.widgets.docker._system import DockerInfo
from terminal.widgets.ItemsList import ItemsList

class DockerContainersList(Widget):
    def __init__(self, go_to, containers: list) -> None:
        self.go_to = go_to
        self.containers = containers
        super().__init__()

    def compose(self) -> ComposeResult:
        yield ContainerTree('Containers', id='containers-tree', classes='with_title')

    def on_mount(self):
        containers_tree = self.query_one('#tree', Tree)
        containers_tree.border_title = ":evergreen_tree: Containers tree"
        # self.containers['nodes'][0]['node'] = containers_tree.root.add('Running Containers', expand=True)
        
        # containers_table.add_column("ID")
        # containers_table.add_column("Image")
        # containers_table.add_column("Time")
        # for container in self.containers:
        #     valid_container = inspect_container(container.id)

    # json.dumps(vars(docker.container.inspect('a9bbcebf6aee').__dict__['_inspect_result']), sort_keys=True, default=str)       

class DockerSideBar(Widget):
    def __init__(self, images: list, volumes: list) -> None:
        self.images = images
        self.volumes = volumes
        super().__init__()

    def compose(self) -> ComposeResult:
        with Container(id="list-grid"):
            yield DataTable(id='images-table', classes='with_title', show_header=True)
            yield DataTable(id='volumes-table', classes='with_title', show_header=True)

    def on_mount(self):
        images_table = self.query_one(f"#images-table", DataTable)
        images_table.border_title = f":horizontal_traffic_light: Images list"
        images_table.add_column("ID")
        images_table.add_column("Tags")
        images_table.add_column("Time")
        for image in self.images:
            valid_image = inspect_image(image.id)
            images_table.add_row(
                Text(valid_image.id.split("sha256:")[1][:8], style=""),
                Text(", ".join(valid_image.__dict__['_inspect_result'].repo_tags) if len(valid_image.__dict__['_inspect_result'].repo_tags) > 0 else 'No tags', style=""),
                Text(valid_image.created.strftime("%d/%m/%Y, %H:%M:%S"), style=""),
            )

        volumes_table = self.query_one(f"#volumes-table", DataTable)
        volumes_table.border_title = f":horizontal_traffic_light: Volumes list"
        volumes_table.add_column("Name")
        volumes_table.add_column("Time")
        for volume in self.volumes:
            volumes_table.add_row(
                Text(volume.name, style=""),
                Text(volume.created_at.strftime("%d/%m/%Y, %H:%M:%S"), style=""),
            )
    
class DockerTab(Container):
    """Screen displaying the loaded data through various widgets."""
    CSS_PATH = "../css/main_layout.tcss"

    def __init__(self, args: dict = {}, data: dict = {}) -> None:
        self.args = args
        self.data = data
        self.loaded = False
        super().__init__(id="tab-docker")

    def compose(self) -> ComposeResult:
        # if not self.loaded:
        #     yield Static(f"{self.data.keys()}")
        #     yield LoadingIndicator()
        # else:
        with Container(id="app-grid"):
            with Horizontal(id="top"):
                Text('docker-test', style="")
            cnt = Container(id="info-grid", classes="titled-container")
            cnt.border_title = "[b]Info[/]"
            with cnt:
                yield DockerInfo()
            # with VerticalScroll(id="main"):
            #     yield ItemsList(self.data['containers'], "Containers")
            # with VerticalScroll(id="list"):
            #     yield DockerSideBar(self.data['images'], self.data['volumes'])

    def on_mount(self):
        self.title = "OsCli"
        self.sub_title = "Docker"
        self.data['images'] = get_images()
        self.data['containers'] = get_containers()
        self.data['volumes'] = get_volumes()
        # self.loaded = True
        # self.recompose()

        # self.build['title']