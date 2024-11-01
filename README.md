# Neural Style Transfer

> Create **artificial artwork** by transferring the appearance of one image (ie a famous painting) to another user-supplied image (e.g. your favorite photo).

**Documentation**: [https://boromir674.github.io/neural-style-transfer/](https://boromir674.github.io/neural-style-transfer/)

<div style="display: flex; align-items: center; margin-bottom: 25px;">
    <img src="./tests/data/canoe_water_w300-h225.jpg" alt="Demo Content Image" style="vertical-align: middle;">
    <span style="margin: 0 10px; font-size: 24px; vertical-align: middle;">+</span>
    <img src="./tests/data/blue-red_w300-h225.jpg" alt="Demo Style Image" style="vertical-align: middle;">
    <span style="margin: 0 10px; font-size: 24px; vertical-align: middle;">=</span>
    <img src="./tests/data/canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100-demo-gui-run-1.png" alt="Demo Gen Image" style="vertical-align: middle;">
</div>

Uses a `Neural Style Transfer` algorithm to transfer the **style** (aka appearance), from one image, into the **contents** of another.


`Neural Style Transfer (NST)` is an algorithm that applies the `style` of an image to the `contents` of another and produces a `generated` image.  
The idea is to find out how someone, with the `painting style` shown in one image, would depict the `contents` shown in another image.

NST takes as INPUT a `Content` image (e.g. a picture taken with your camera) and a `Style` image (e.g. a Van Gogh painting) and `Generates` a new image.



# Overview
This project provides an `NST` algorithm through
- the `artificial_artwork` **Python package** (aka module)
- the `nst` **CLI**
- the `boromir674/neural-style-transfer` **Docker image**

<!-- This package exposes a configurable NST algorithm via a convenient CLI program. -->

| Build | Package | Containerization | Code Quality |
|-------|---------|------------------|--------------|
| ![CI Pipeline Status][ci-pipeline-status] | ![PyPI][pypi] ![Wheel][wheel] ![Python Versions][python-versions] ![Commits Since][commits-since] | ![Docker][docker] ![Image Size][image-size] | ![Codacy][codacy] ![Code Climate][code-climate] ![Maintainability][maintainability] ![Scrutinizer][scrutinizer] |


## Features
- VGG-19 Convolutional Neural Network, as `Image model`
- Selection of `style layers` at runtime
- Efficient Iterative Learning Algorithm, with `tensorflow`
- Fast minimization of loss/cost function with parallel/multicore execution
- Selection of Algorithm Termination Condition/Criteria, at runtime
- Periodic persistance of `Generated image`, during Learning loop

## Quick-start

Run a demo NST, on sample `Content` and `Style` Images:

```sh
mkdir art
export NST_HOST_MOUNT="$PWD/art"

docker-compose up

# Process runs, in containerized environment, and exits.
```

Check out your Generated Image! Artificial Artwork: art/canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100.png

```shell
xdg-open art/canoe_water_w300-h225.jpg+blue-red_w300-h225.jpg-100.png
```

## Usage

Run the `nst` CLI with the `--help` option to see the available options.

```shell
docker run boromir674/neural-style-transfer:1.0.2 --help
```

## Development

### Installation

Install `nst` CLI and `artificial_artwork` python package from `pypi``:

Note: Installation on Debian-based Distros for Python 3.11 require `distutils` which is
not included in python3.11 standard distribution (but included in python3.10).

```shell
sudo apt-get install python3.11-distutils
```

```shell
pip install artificial_artwork
```

Only python3.8 wheel is included atm.


Sample commands to install the NST CLI from source, using a terminal:

```shell
git clone https://github.com/boromir674/neural-style-transfer.git
    
pip install ./neural-style-transfer
```

> The Neural Style Transfer - CLI heavely depends on Tensorflow (tf) and therefore it is crucial that tf is installed correctly in your Python environment.


[ci-pipeline-status]: https://img.shields.io/github/actions/workflow/status/boromir674/neural-style-transfer/test.yaml?branch=master&label=build&logo=github-actions&logoColor=233392FF
[pypi]: https://img.shields.io/pypi/v/artificial-artwork?color=blue&label=pypi&logo=pypi&logoColor=%23849ed9
[wheel]: https://img.shields.io/pypi/wheel/artificial-artwork?logo=python&logoColor=%23849ed9
[python-versions]: https://img.shields.io/pypi/pyversions/artificial-artwork?color=blue&logo=python&logoColor=%23849ed9
[commits-since]: https://img.shields.io/github/commits-since/boromir674/neural-style-transfer/v1.0.1/master?color=blue&logo=Github
[docker]: https://img.shields.io/docker/v/boromir674/neural-style-transfer/latest?logo=docker&logoColor=%23849ED9
[image-size]: https://img.shields.io/docker/image-size/boromir674/neural-style-transfer/latest?logo=docker&logoColor=%23849ED9
[codacy]: https://app.codacy.com/project/badge/Grade/07b27ac547a94708aefc5e845d2b6d01
[code-climate]: https://api.codeclimate.com/v1/badges/2ea98633f88b75e87d1a/maintainability
[maintainability]: https://img.shields.io/codeclimate/tech-debt/boromir674/neural-style-transfer?logo=CodeClimate
[scrutinizer]: https://img.shields.io/scrutinizer/quality/g/boromir674/neural-style-transfer/master?logo=scrutinizer-ci

