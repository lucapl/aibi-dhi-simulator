import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

from environment.profiled_fogg_model import PROFILES


lock = threading.Lock()

outputs = {}   # profile -> list[str]
status = {}    # profile -> "running" | "done" | "error"

MAX_LINES = 5


def append_line(profile, line):
    lines = outputs[profile]
    lines.append(line)

    if len(lines) > MAX_LINES:
        del lines[:len(lines) - MAX_LINES]


def run_cmd(profile, cmd):
    with lock:
        outputs[profile] = [f"Starting {profile}..."]
        status[profile] = "running"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        with lock:
            append_line(profile, line.rstrip())

    process.wait()

    with lock:
        if process.returncode == 0:
            status[profile] = "done"
            append_line(profile, f"[exit 0]")
        else:
            status[profile] = "error"
            append_line(profile, f"[exit {process.returncode}]")


def render():
    layout = Layout()

    panels = []
    with lock:
        for p in outputs:
            text = Text("\n".join(outputs[p]))

            if status[p] == "running":
                style = "yellow"
            elif status[p] == "done":
                style = "green"
            else:
                style = "red"

            panels.append(
                Panel(text, title=p, border_style=style)
            )

    layout.split_column(*panels)
    return layout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--profiles", nargs='*', default=list(PROFILES.keys()))
    parser.add_argument("--results_directory", type=str, default='results')
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--notebook_path", type=str, default='experiments.ipynb')
    args = parser.parse_args()

    commands = []
    for profile in args.profiles:
        res_dir = (Path(args.results_directory) / profile).as_posix()
        params = dict(
            runs=args.runs,
            profile=profile,
            results_directory=res_dir
        )

        command = (
            ['papermill', args.notebook_path, f'experiment_profile_{profile}.ipynb']
            + sum([["-p", k, str(v)] for k, v in params.items()], [])
        )

        commands.append((profile, command))

    with Live(render(), refresh_per_second=10) as live:

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_cmd, p, c) for p, c in commands]

            while any(not f.done() for f in futures):
                live.update(render())

            live.update(render())