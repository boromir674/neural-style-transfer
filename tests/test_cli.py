import typing as t

import pytest


@pytest.mark.runner_setup(mix_stderr=False)
def test_cli_demo(test_suite, isolated_cli_runner, monkeypatch):
    from pathlib import Path
    from artificial_artwork.cli import entry_point as main
    from artificial_artwork import _demo
    monkeypatch.setattr(_demo, 'source_root_dir', Path(test_suite) / '..')
    result = isolated_cli_runner.invoke(
        main,
        # args=['demo', '--help'],
        args=['demo', '-it', '4'],
        input=None,
        env=None,
        catch_exceptions=False,
        color=False,
        # **kwargs,
    )
    assert result.exit_code == 0


@pytest.mark.runner_setup(mix_stderr=False)
def test_cli_main(test_suite, isolated_cli_runner, monkeypatch):
    from pathlib import Path
    from artificial_artwork.cli import entry_point as main
    from artificial_artwork import _demo
    # monkeypatch.setattr(_demo, 'source_root_dir', Path(test_suite) / '..')
    result = isolated_cli_runner.invoke(
        main,
        # args=['demo', '--help'],
        args=[
            'run',
            str(Path(test_suite) / 'data' / 'canoe_water_w300-h225.jpg'),
            str(Path(test_suite) / 'data' / 'blue-red_w300-h225.jpg'),
            '--iterations',
            '6',
            '--location',  # output folder to store snapshots of Gen Image
            '/tmp',
        ],
        input=None,
        env=None,
        catch_exceptions=False,
        color=False,
        # **kwargs,
    )
    assert result.exit_code == 0
