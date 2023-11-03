import os
import sys
import typing as t

import click

from artificial_artwork import __version__

from ._main import create_algo_runner
from .cmd_demo import demo


class WithStateAttribute(t.Protocol):
    """Protocol for classes that have a state attribute."""

    state: t.Any


class HandleAlgorithmProgressUpdatesAble(t.Protocol):
    def update(self, subject: WithStateAttribute) -> None:
        ...


this_file_location = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))


# CLI --version flag
def version_msg():
    """artificial_artwork CLI version, lib location and Python version.

    Get message about artificial_artwork version, location
    and Python version.
    """
    # extract everything about version: major, minor, patch and build notes
    python_version = sys.version
    message = "Neural Style Transfer CLI %(version)s from {} (Python {})"
    location = os.path.dirname(this_file_location)
    return message.format(location, python_version)


# MAIN
@click.group()
@click.version_option(__version__, "-V", "--version", message=version_msg())
def entry_point():
    pass


# RUN CMD
@click.command()
@click.argument("content_image")
@click.argument("style_image")
@click.option("--iterations", "-it", type=int, default=100, show_default=True)
@click.option("--location", "-l", type=str, default=".")
def run(content_image, style_image, iterations, location):
    backend_objs: t.Dict[str, t.Any] = create_algo_runner(
        iterations=iterations,
        output_folder=location,
        noisy_ratio=0.6,
    )
    run_nst: t.Callable[[str, str], None] = backend_objs["run"]
    # subscribe_callback: t.Callable[[HandleAlgorithmProgressUpdatesAble], None] = backend_objs['subscribe']

    run_nst(content_image, style_image)


### NST CLI Entrypoint ###

# ATTACH CMDs

# 1st sub command: `nst run`
entry_point.add_command(run)

# 2nd sub command: `nst demo`
entry_point.add_command(demo)
