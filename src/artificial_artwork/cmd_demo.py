import os
import typing as t
from pathlib import Path

import click

from ._demo import create_algo_runner as init_algo_infra


class WithStateAttribute(t.Protocol):
    """Protocol for classes that have a state attribute."""

    state: t.Any


class HandleAlgorithmProgressUpdatesAble(t.Protocol):
    def update(self, subject: WithStateAttribute) -> None:
        ...


# Ask user what to do depending on whether the output folder already exists
def validate_and_normalize_path(ctx, param, value):
    """Custom function to validate and normalize a path."""
    if value is None:
        return None
    path = Path(value)

    if path.is_absolute():
        abs_path = path
    else:
        current_directory = Path.cwd()
        abs_path = current_directory / path

    if not abs_path.exists():
        abs_path.mkdir()
        click.echo(f'Folder "{abs_path}" created')
    else:
        # get files inside the folder
        folder_files = [f for f in abs_path.iterdir() if f.is_file()]
        if len(folder_files) > 0:
            # ask user whether to delete everything, process as it is or exit
            click.echo(f'Folder "{abs_path}" already exists and is not empty.')
            click.echo("What do you want to do?")
            click.echo("1. Delete everything and start from scratch")
            click.echo("2. Process the existing files")
            click.echo("3. Exit")
            choice = click.prompt("Enter your choice", type=int)
            if choice == 1:
                click.echo("Deleting everything...")
                for file in folder_files:
                    file.unlink()
            elif choice == 2:
                click.echo("Processing existing files...")
            elif choice == 3:
                click.echo("Exiting...")
                ctx.exit()
            else:
                raise click.BadParameter(f'Invalid choice "{choice}".')
    return abs_path


# DEMO CMD
@click.command()
@click.option(
    "-it",
    "--iterations",
    type=int,
    default=100,
    show_default=True,
    help="Number of iterations to run the algorithm.",
)
@click.option(
    "-o",
    "--output",
    "output_folder",
    # type=click.Path(exists=True),
    type=click.Path(
        # exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    default="demo-output",
    show_default=True,
    help="Location to save the generated images.",
    callback=validate_and_normalize_path,
)
def demo(iterations, output_folder):
    print("[DEBUG] output type: {}".format(type(output_folder)))
    click.echo(f"Running demo with {iterations} iterations and location {output_folder}.")

    # By default the backend adds 2 Listeners/Observers to the Subject/State object
    # The Subject being an object representing the current state of the Algorithm
    # Progress (ie current Cost Values for Jc, Js, Jt, current iteration, etc)
    # The backend does it by suscribing the Listeners to the Subject (see below
    # the subscribe_to_algorithm_progress method in case you want to add more)

    # One Observer configured by the backend receives updates more frequently
    # and it prints the Progress stats/metrics as a formatted message to the
    # console

    # The other Observer configured by the backend receives updates less frequently
    # and it saves a snapshot of the Generated image during the current iteration
    # to the output folder
    backend_objects = init_algo_infra(
        iterations=iterations,
        output_folder=output_folder,
        # if None then default works only on editable installation
        # ie if code was install with `pip install -e`, or if was added manually to PATH
        content_img_file=os.environ.get('CONTENT_IMAGE_DEMO'),
        style_img_file=os.environ.get('STYLE_IMAGE_DEMO'),
    )

    # destructuring the backend objects
    run_algorithm: t.Callable[[], None] = backend_objects["run"]
    # subscribe_to_algorithm_progress: t.Callable[[HandleAlgorithmProgressUpdatesAble], None] = \
    #     backend_objects['subscribe']

    run_algorithm()

    # print a stylized message to console informing that program finished suuccessfully :)

    click.secho("Demo finished successfull :)", fg="green", bold=True)
    click.secho("Check the output folder for the generated images.", fg="green", bold=True)
    click.secho("xdg-open {}".format(output_folder), fg="green", bold=True)
    click.secho("Bye!", fg="green", bold=True)
