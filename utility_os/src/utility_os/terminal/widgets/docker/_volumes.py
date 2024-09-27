# ---- Docker Volumes Widget

## --- Libs
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable

class DockerVolumes(Widget):
    def __init__(self, volumes: list) -> None:
        self.volumes = volumes
        super().__init__()

    def compose(self) -> ComposeResult:
        yield DataTable(id='volumes-list', classes='with_title', show_header=True)

    def on_mount(self):
        volumes_table = self.query_one(f"#volumes-list", DataTable)
        volumes_table.border_title = f":horizontal_traffic_light: Volumes list"
        volumes_table.add_column("Name")
        volumes_table.add_column("Time")
        for volume in self.volumes:
            volumes_table.add_row(
                Text(volume.name, style=""),
                Text(volume.created_at.strftime("%d/%m/%Y, %H:%M:%S"), style=""),
            )