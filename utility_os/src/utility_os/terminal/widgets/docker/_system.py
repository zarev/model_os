# ---- Docker System Widget

## --- Libs
from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable
from textual.containers import Container, Horizontal

## --- Locals
### -- Api & utils
from api.docker import get_images, get_containers, get_volumes, get_system, inspect_image, inspect_container
from utils.bytes import humanbytes

class DockerInfo(Widget):
    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        with Horizontal(id = "info-tab"):
            yield DataTable(id='info-table', classes='with_title', show_header=False)
            yield DataTable(id='system-table', classes='with_title', show_header=False)
            yield DataTable(id='disk-table', classes='with_title', show_header=False)

    def on_mount(self):
        self.data = get_system()
        info_table = self.query_one('#info-table', DataTable)
        info_table.border_title = f":computer: Resources"
        info_table.add_column("Param")
        info_table.add_column("Value")
        info_table.add_row(
            Text('Available CPU', style=""),
            Text(f"{self.data['info'].n_cpu} cores", style=""),
        )
        info_table.add_row(
            Text('Available MEM', style=""),
            Text(humanbytes(self.data['info'].mem_total), style=""),
        )
        info_table.add_row(
            Text('Architecture', style=""),
            Text(self.data['info'].architecture, style=""),
        )
        info_table.add_row(
            Text('Docker version', style=""),
            Text(f"{self.data['info'].server_version} ({self.data['info'].operating_system})", style=""),
        )
        info_table.add_row(
            Text('Kernel in use', style=""),
            Text(self.data['info'].kernel_version, style=""),
        )

        system_table = self.query_one('#system-table', DataTable)
        system_table.border_title = f":whale: System info"
        system_table.add_column("Param")
        system_table.add_column("Value")
        system_table.add_row(
            Text('Images', style=""),
            Text(str(self.data['info'].images), style=""),
        )
        system_table.add_row(
            Text('Containers', style=""),
            Text(f"{self.data['info'].containers_running} running / {self.data['info'].containers_stopped} stopped ({self.data['info'].containers} total)", style=""),
        )
        system_table.add_row(
            Text('Plugins (Network)', style=""),
            Text(", ".join(self.data['info'].plugins.network), style=""),
        )
        system_table.add_row(
            Text('Plugins (Volume)', style=""),
            Text(", ".join(self.data['info'].plugins.volume), style=""),
        )
        system_table.add_row(
            Text('Plugins (Logging)', style=""),
            Text(", ".join(self.data['info'].plugins.log), style=""),
        )

        disk_table = self.query_one('#disk-table', DataTable)
        disk_table.border_title = f":floppy_disk: Disk info"
        disk_table.add_column("Param")
        disk_table.add_column("Value")
        disk_table.add_row(
            Text('Active images', style=""),
            Text(f"{self.data['disk'].images.active}", style=""),
        )
        disk_table.add_row(
            Text('Reclaimable space from containers', style=""),
            Text(f"{humanbytes(self.data['disk'].containers.reclaimable)}", style=""),
        )
        disk_table.add_row(
            Text('Reclaimable space from volumes', style=""),
            Text(f"{self.data['disk'].volumes.reclaimable_percent}%", style=""),
        )
        disk_table.add_row(
            Text('Number of builds caches', style=""),
            Text(f"{self.data['disk'].build_cache.total_count}", style=""),
        )
        disk_table.add_row(
            Text('Size of builds caches', style=""),
            Text(f"{humanbytes(self.data['disk'].build_cache.size)}", style=""),
        )
