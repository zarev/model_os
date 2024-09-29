# Libs
from textual.app import ComposeResult
from textual.widgets import Static, DataTable, ListItem, ListView
from textual.widget import Widget
# Locals
from utils.time import convert_millis, to_ISO

class BuildStatusTable(Widget):

    def __init__(self, build: dict) -> None:
        self.build = build
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(id='summary', classes='with_title')
        yield DataTable(id='status-table', classes='with_title', show_header=False)
        yield DataTable(id='timings-table', classes='with_title', show_header=False)
        yield ListView(id='changes-list', classes='with_title')
            
    def on_mount(self):
        summary_text = self.query_one('#summary')
        summary_text.border_title = ":chequered_flag: Summary"
        summary_text.update(self.build['summary'])

        status_table = self.query_one('#status-table', DataTable)
        status_table.border_title = ":vertical_traffic_light: Results"
        status_table.add_column("key")
        status_table.add_column("value")
        status_table.add_row("Result", f":{self.build['status']}_circle: {self.build['data']['result'].capitalize()}")
        status_table.add_row("Author", self.build['data']['causes'][0].get('userId'))

        timings_table = self.query_one('#timings-table', DataTable)
        timings_table.border_title = ":timer_clock:  Timings"
        timings_table.add_column("key")
        timings_table.add_column("value")
        start_time = to_ISO(self.build['data']['startTime'].split('+')[0]).strftime("%x %X")
        timings_table.add_row("Start", start_time)
        end_time = to_ISO(self.build['data']['endTime'].split('+')[0]).strftime("%x %X")
        timings_table.add_row("End", end_time)
        duration = convert_millis(int(self.build['data']['durationInMillis']))
        timings_table.add_row("Duration", f"approx {duration}")
        
        changes_list = self.query_one('#changes-list', ListView)
        changes_list.border_title = ":page_facing_up: Changeset"
        for change in self.build['data']['changeSet']:
            changes_list.append(ListItem(
                Static(f"[cyan bold]{change['commitId'][:8]}[not cyan][not bold] - {change['msg']}")
            ))