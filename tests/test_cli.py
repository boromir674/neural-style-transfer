import typing as t

import pytest


@pytest.mark.runner_setup(mix_stderr=False)
def test_cli_demo(test_suite, toy_nst_algorithm, isolated_cli_runner, monkeypatch):
    """Verify process exits with 0 after calling the CLI as `nst demo -it 4`.

    Test that verifies the process exits with 0 when the CLI is invoked as
    `nst demo -it 4`.

    This means the NST receives as input Content and Style Images the 2 images
    shipped with the Source Distribution for demoing purposes.

    The NST is expected to iterate/learn (number of epochs to run) for 4 times.

    The process is run in isolation, meaning that the process' stdout and
    stderr are not mixed with the pytest's stdout and stderr.
    """
    from pathlib import Path

    from artificial_artwork import _demo
    from artificial_artwork.cli import entry_point as main

    # monkey patch _demo module to trick the _demo module in believing it is
    # inside the Test Suite dir ('tests/'), so that it properly locates the demo
    # Content and Style Images
    monkeypatch.setattr(_demo, "source_root_dir", Path(test_suite) / "..")

    # Defer from using Production Pretrained Weights, and instead use the Toy Network
    # That way this Test Case runs as a Unit Test, and does not need to integrate
    # with the Production VGG Image Model.
    # We achieve that by monkeypatching at runtime all the necessary objects, so that
    # the program uses the Toy Network, which has but 1 Conv Layer (with very small
    # dimensions too), with weights to use for the NST (as pretrained weights)
    toy_nst_algorithm()  # use fixture callable, which leverages monkeypatch under the hood

    # Call CLI as `nst demo -it 4` in isolation
    result = isolated_cli_runner.invoke(
        main,
        args=["demo", "-it", "4"],
        input=None,
        env=None,
        catch_exceptions=False,
        color=False,
        # **kwargs,
    )
    assert result.exit_code == 0
    # GIVEN we can capture the stdout of the CLI (ie as a User would see if
    # calling the CLI in an interactive shell)
    assert type(result.stdout) == str

    # WHEN we inspect the stdout of the CLI
    string_to_inspect = result.stdout

    # THEN we expect to see the following: VGG Mat Weights Mock Loader Called 1 time
    # (ie the CLI called the VGG Mat Weights Mock Loader 1 time)
    exp_str = "VGG Mat Weights Mock Loader Called"

    stdout_lines: t.List[str] = string_to_inspect.split("\n")
    exp_str_appearances = stdout_lines.count(exp_str)
    assert exp_str_appearances == 1


@pytest.mark.runner_setup(mix_stderr=False)
def test_cli_main(test_suite, toy_nst_algorithm, isolated_cli_runner):
    from pathlib import Path

    from artificial_artwork.cli import entry_point as main

    # Monkey Patch Prod NST (Prod Pretrained Weights) to use Toy Network (Toy Pretrained Weights)
    toy_nst_algorithm()  # use fixture callable, which leverages monkeypatch under the hood

    result = isolated_cli_runner.invoke(
        main,
        args=[
            "run",
            str(Path(test_suite) / "data" / "canoe_water_w300-h225.jpg"),
            str(Path(test_suite) / "data" / "blue-red_w300-h225.jpg"),
            "--iterations",
            "6",
            "--location",  # output folder to store snapshots of Gen Image
            "/tmp",  # TODO use os native pytest fixture for tempdir
        ],
        input=None,
        env=None,
        catch_exceptions=False,
        color=False,
        # **kwargs,
    )
    assert result.exit_code == 0
