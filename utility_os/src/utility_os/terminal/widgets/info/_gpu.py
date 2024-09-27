import GPUtil
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.widget import Widget

from utils.generic import sizeof_fmt
from utils.braille_stream import BrailleStream

colors = ["yellow", "green", "blue", "magenta", "red"]

class GPU(Widget):
    def on_mount(self):
        gpu = GPUtil.getGPUs()[0]
        self.mem_total = gpu.memoryTotal

        # check which mem sections are available on the machine
        self.attrs = []
        for attr in ["memoryUtil", "memoryUsed", "memoryFree"]:
            if hasattr(gpu, attr):
                self.attrs.append(attr)

        # append spaces to make all names equally long
        maxlen = max(len(string) for string in self.attrs)
        maxlen = min(maxlen, 6)
        self.labels = [attr[maxlen:].ljust(maxlen) for attr in self.attrs]

        # can't use
        # [BrailleStream(40, 4, 0.0, self.mem_total_bytes)] * len(self.names)
        # since that only creates one BrailleStream, references n times.
        self.mem_streams = []
        for attr in self.attrs:
            self.mem_streams.append(BrailleStream(40, 4, 0.0, self.mem_total))

        self.group = Group("", "", "")
        self.panel = Panel(
            self.group,
            title=f"[b]GPU[/] - {gpu.name} ({gpu.driver})",
            title_align="left",
            border_style="white",
            box=box.SQUARE,
        )

        self.refresh_table()
        self.set_interval(2.0, self.refresh_table)

    def refresh_table(self):
        gpu = GPUtil.getGPUs()[0]

        for k, (attr, label, stream, col) in enumerate(
            zip(self.attrs, self.labels, self.mem_streams, colors)
        ):
            val = getattr(gpu, attr)
            total = self.mem_total

            stream.add_value(val)
            val_string = " ".join(
                [
                    label,
                    sizeof_fmt(val*1000000, fmt=".2f"),
                    f"({val / total * 100:.0f}%)",
                ]
            )
            graph = "\n".join(
                [val_string + stream.graph[0][len(val_string) :]] + stream.graph[1:]
            )
            self.group.renderables[k] = Text(graph, style=col)

        self.refresh()

    def render(self) -> Panel:
        return self.panel

    async def on_resize(self, event):
        for ms in self.mem_streams:
            ms.reset_width(event.size.width - 4)

        # split the available event.height-2 into n even blocks, and if there's
        # a rest, divide it up into the first, e.g., with n=4
        # 17 -> 5, 4, 4, 4
        n = len(self.attrs)
        heights = [(event.size.height - 2) // n] * n
        for k in range((event.size.height - 2) % n):
            heights[k] += 1
            # add to last:
            # heights[-(k + 1)] += 1

        for ms, h in zip(self.mem_streams, heights):
            ms.reset_height(h)

        self.refresh_table()