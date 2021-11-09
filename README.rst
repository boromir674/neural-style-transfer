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

Sample commands to install NST CLI using `bash`

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


Usage
-----
A sample example is below.


::

    neural-style-transfer tests/data/canoe_water.jpg van-gogh.jpg


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

