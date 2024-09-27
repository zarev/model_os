# Libs
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import DataTable
# Locals
from utils.time import convert_millis

class ItemsList(Widget):
    CSS_PATH = "css/main_layout.tcss"
    
    def __init__(self, items: list, title: str) -> None:
        self.items = items
        self.title = title
        super().__init__()

    def compose(self) -> ComposeResult:
        yield DataTable(id=f"{self.title}-list-table", classes='with_title', show_header=True)

    def on_mount(self):
        items_table = self.query_one(f"#{self.title}-list-table", DataTable)
        items_table.border_title = f":horizontal_traffic_light: {self.title} list"
        items_table.cursor_type = 'row'
        # items_table.add_column("Uptime")
        #     duration = convert_millis(int(build['durationInMillis']))
        if(self.title == 'Images'):
            items_table.add_column("ID")
            items_table.add_column("Tags")
            items_table.add_column("Time")
            for item in self.items:
                items_table.add_row(
                    Text(
                        item.id.split("sha256:")[1][:8],
                        style="white on blue",
                    ),
                    Text(
                        item._inspect_result.repo_tags[0] if item._inspect_result != None else '',
                        style="white on blue",
                    ),
                    Text(
                        item.created.strftime("%m/%d/%Y, %H:%M:%S"),
                        style="white on blue",
                    ),
                )
            
        elif(self.title == 'Volumes'):
            items_table.add_column("name")
            items_table.add_column("time")
            for item in self.items:
                items_table.add_row(
                    Text(
                        item.name,
                        style="white on blue",
                    ),
                    Text(
                        item.created_at.strftime("%m/%d/%Y, %H:%M:%S"),
                        style="white on blue",
                    ),
                )
            

    def on_data_table_row_selected(self, event):
        self.items['id'] = event.row_key.value
