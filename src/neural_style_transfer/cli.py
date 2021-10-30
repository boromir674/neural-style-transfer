
import os
import click

from .disk_operations import Disk
from .styling_observer import StylingObserver
from .algorithm import NSTAlgorithm
from .runner import Runner


@click.argument('content_image')
@click.argument('style_image')
@click.option('--interactive', '-i', type=bool, default=True)
@click.option('--location', '-l', type=str, default='')
def cli(content_image, style_image, interactive, location):
    styling_observer = StylingObserver(location)
    algorithm = NSTAlgorithm(iterations=10)
    runner = Runner(algorithm, disk=Disk)
    runner.attach_observer(styling_observer)
    runner.run(content_image, style_image, interactive=interactive)


if __name__ == '__main__':
    cli()
