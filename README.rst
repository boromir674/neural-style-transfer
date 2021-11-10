Neural Style Transfer - CLI
===========================

`Neural Style Tranfer` (NST) is an algorithm that applies the `style` of an image to the `contents` of another and produces a `generated` image.
The idea is to find out how someone, with the `painting style` shown in one image, would depict the `contents` shown in another image.

NST takes a `content` image (eg picture taken with your camera) and a `style` image (eg a picture of a Van Gogh painting) and produces the `generated` image.

This Python package runs a Neural Style Tranfer algorithm on input `content` and `style` images to produce `generated` images.


.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |circleci| |codecov|


    * - code quality
      - |better_code_hub| |code_climate| |maintainability| |codacy| |scrutinizer|



Overview
========

This package exposes a configurable NST algorithm via a convenient CLI program.

Key features of the package:

* Selection of style layers at runtime
* Iterative Learning Algorithm using the VGG Deep Neural Network
* Selection of iteration termination condition at runtime
* Fast minimization of loss/cost function with parallel/multicore execution, using Tensorflow
* Persisting of generated images



Installation
------------
| The Neural Style Transfer - CLI heavely depends on Tensorflow (tf) and therefor it is crucial that tf is installed correctly in your Python environment.

Sample commands to install NST CLI using a terminal:

::

    # Get the Code
    git clone https://github.com/boromir674/neural-style-transfer.git
    cd neural-style-transfer

    # Activate a python virtual environment
    virtualenv env --python=python3
    source env/bin/activate

    # Install dependencies
    pip install -r requirements/dex.txt

    # Install NST CLI (in virtual environment)
    pip install -e .

    # Setup a symbolic link (in your host system) in a location in your PATH
    # Assuming ~/.local/bin is in your PATH
    ln -s $PWD/env/bin/neural-style-transfer ~/.local/bin/neural-style-transfer

    # Deactivate environment since the symbolic link is available in "global scope" by now
    deactivate


Usage
-----

Download the Vgg-Verydeep-19 pretrained `model` from https://mega.nz/file/i5xDWI4Y.

Exctract the model (weights and layer architecture).

For example use `tar -xvf imagenet-vgg-verydeep-19.tar` to extract in the current directory.

Indicate to the program where to find the model:

::

    export AA_VGG_19=$PWD/imagenet-vgg-verydeep-19.mat

We have included one 'content' and one 'style' image in the source repository, to facilitate testing.
You can use these images to quickly try running the program.

For example, you can get the code with `git clone git@github.com:boromir674/neural-style-transfer.git`,
then `cd neural-style-transfer`.

Assuming you have installed using a symbolic link in your PATH (as shown above), or if you are still
operating withing your virtual environment, then you can create artificial artwork with the following command.

The algorithm will apply the style to the content iteratively.
It will iterate 100 times. 

::

    # Create a directory where to store the artificial artwork
    mkdir nst_output

    # Run a Neural Style Algorithm for 100 iterations and store output to nst_output directory
    neural-style-transfer tests/data/canoe_water.jpg tests/data/blue-red-w400-h300.jpg --location nst_output


Note we are using as 'content' and 'style' images jpg files included in the distribution (artificial-artwork package).
We are using a photo of a canoe on water and an abstract painting with prevalence of blue and red color shades.

Also note that to demonstrate quicker, both images have been already resized to just 400 pixels of width and 300 of height each.

Navigating to `nst_output` you can find multiple image files generated from running the algorithm. Each file corresponds to the
image generated on a different iteration while running the algorithm. The bigger the iteration the more "style" has been applied.

Check out your artificial artwork!




.. |circleci|  image:: https://img.shields.io/circleci/build/github/boromir674/neural-style-transfer/master?logo=circleci
    :alt: CircleCI
    :target: https://circleci.com/gh/boromir674/neural-style-transfer/tree/master


.. |codecov| image:: https://codecov.io/gh/boromir674/neural-style-transfer/branch/master/graph/badge.svg?token=3POTVNU0L4
    :alt: Codecov
      :target: https://codecov.io/gh/boromir674/neural-style-transfer
    

.. |better_code_hub| image:: https://bettercodehub.com/edge/badge/boromir674/neural-style-transfer?branch=master
    :alt: Better Code Hub
    :target: https://bettercodehub.com/


.. |codacy| image:: https://app.codacy.com/project/badge/Grade/07b27ac547a94708aefc5e845d2b6d01
    :alt: Codacy
    :target: https://www.codacy.com/gh/boromir674/neural-style-transfer/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=boromir674/neural-style-transfer&amp;utm_campaign=Badge_Grade


.. |code_climate| image:: https://api.codeclimate.com/v1/badges/2ea98633f88b75e87d1a/maintainability
   :alt: Maintainability
   :target: https://codeclimate.com/github/boromir674/neural-style-transfer/maintainability


.. |maintainability| image:: https://img.shields.io/codeclimate/tech-debt/boromir674/neural-style-transfer?logo=CodeClimate
    :alt: Technical Debt
    :target: https://codeclimate.com/github/boromir674/neural-style-transfer/maintainability


.. |scrutinizer| image:: https://img.shields.io/scrutinizer/quality/g/boromir674/neural-style-transfer/master?logo=scrutinizer-ci
    :alt: Scrutinizer code quality
    :target: https://scrutinizer-ci.com/g/boromir674/neural-style-transfer/?branch=master



.. |version| image:: https://img.shields.io/pypi/v/topic-modeling-toolkit.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/topic-modeling-toolkit

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/topic-modeling-toolkit.svg
    :alt: Supported versions
    :target: https://pypi.org/project/topic-modeling-toolkit

