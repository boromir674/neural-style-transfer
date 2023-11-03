def test_nst_produces_same_image_when_run_on_different_processes(
    test_suite,
):
    from pathlib import Path

    import imageio
    from numpy.typing import NDArray

    # GIVEN the Generated Image from the NST algorithm run on a single process
    run_1_gen_img_path: Path = (
        Path(test_suite)
        / "data"
        / "canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100-demo-gui-run-1.png"
    )

    # GIVEN the Generated Image from the NST algorithm run on another process
    run_2_gen_img_path: Path = (
        Path(test_suite)
        / "data"
        / "canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100-demo-gui-run-2.png"
    )

    # WHEN loading the images into memory
    array_1: NDArray = imageio.imread(str(run_1_gen_img_path))
    array_2: NDArray = imageio.imread(str(run_2_gen_img_path))

    # WHEN comparing the two images, by comparing their pixel values (arrays)
    all_array_values_are_equal: bool = (array_1 == array_2).all()

    # THEN the two images should be the same
    assert all_array_values_are_equal
