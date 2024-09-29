import getpass
import platform
import time
from datetime import datetime, timedelta

import distro
import psutil
from textual.widget import Widget
from rich.text import Text


class InfoLine(Widget):
    def on_mount(self):
        self.width = 0
        self.height = 0
        self.set_interval(1.0, self.refresh)

        # The getlogin docs say:
        # > For most purposes, it is more useful to use getpass.getuser() [...]
        # username = os.getlogin()
        username = getpass.getuser()
        ustring = f"{username} @"
        node = platform.node()
        if node:
            ustring += f" [b]{platform.node()}[/]"

        system = platform.system()
        if system == "Linux":
            ri = distro.os_release_info()
            system_list = [ri["name"]]
            if "version_id" in ri:
                system_list.append(ri["version_id"])
            system_list.append(f"{platform.architecture()[0]} / {platform.release()}")
            system_string = " ".join(system_list)
        elif system == "Darwin":
            system_string = f"macOS {platform.mac_ver()[0]}"
        else:
            # fallback
            system_string = ""

        self.left_string = " ".join([ustring, system_string])
        self.boot_time = psutil.boot_time()

    def render(self):
        uptime = timedelta(seconds=time.time() - self.boot_time)
        h, m = seconds_to_h_m(uptime.seconds)

        right = [f"up {uptime.days}d, {h}:{m:02d}h"]

        bat = psutil.sensors_battery()
        if bat is not None:
            # hh, mm = seconds_to_h_m(bat.secsleft)
            bat_string = f"bat {bat.percent:.1f}%"
            if bat.power_plugged:
                bat_string = "[green]" + bat_string + "[/]"
            elif bat.percent < 10:
                bat_string = "[red reverse bold]" + bat_string + "[/]"
            elif bat.percent < 15:
                bat_string = "[red]" + bat_string + "[/]"
            elif bat.percent < 20:
                bat_string = "[yellow]" + bat_string + "[/]"
            right.append(bat_string)

        return Text(str(self.left_string + ", ".join(right)))

    async def on_resize(self, event):
        self.width = event.size.width
        self.height = event.size.height


def seconds_to_h_m(seconds):
    return seconds // 3600, (seconds // 60) % 60