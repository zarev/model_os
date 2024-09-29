# ---- Docker Images Widget

## --- Libs
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable

## --- Locals
### -- Api & utils
from api.docker import inspect_image, get_images

class DockerImagesTable(Widget):
    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        yield DataTable(id='images-list', classes='with_title', show_header=True)

    def on_mount(self):
        self.images = get_images()
        images_table = self.query_one(f"#images-list", DataTable)
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