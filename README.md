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
      - | |travis|
        | |coverage|
        | |scrutinizer_code_quality|
        | |code_intelligence|
    * - package
      - |version| |python_versions|


========
Overview
========

This package exposes a configurable NST algorithm via a convenient CLI program.

Key features of the package:

* Selection of style layers at runtime
* Iterative Learning Algorithm using the VGG Deep Neural Network
* Selection of iteration termination condition at runtime
* Fast minimization of loss/cost function with parallel/multicore execution, using Tensorflow
* Persisting of generated images


.. _BigARTM: https://github.com/bigartm


Installation
------------
| The Neural Style Transfer - CLI heavely depends on Tensorflow (tf) and therefor it is crucial that tf is installed correctly in your Python environment.

Sample commands to isntall NST CLI using `bash`

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


| either by following the instructions `here <https://bigartm.readthedocs.io/en/stable/installation/index.html>`_ or by using
| the 'build_artm.sh' script provided. For example, for python3 you can use the following


Usage
-----
A sample example is below.


::

    neural-style-transfer tests/data/canoe_water.jpg van-gogh.jpg


Citation
--------

1. Vorontsov, K. and Potapenko, A. (2015). `Additive regularization of topic models <http://machinelearning.ru/wiki/images/4/47/Voron14mlj.pdf>`_. Machine Learning, 101(1):303–323.




.. |travis| image:: https://travis-ci.org/boromir674/topic-modeling-toolkit.svg?branch=dev
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/boromir674/topic-modeling-toolkit

.. |coverage| image:: https://img.shields.io/codecov/c/github/boromir674/topic-modeling-toolkit/dev?style=flat-square
    :alt: Coverage Status
    :target: https://codecov.io/gh/boromir674/topic-modeling-toolkit/branch/dev

.. |scrutinizer_code_quality| image:: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/badges/quality-score.png?b=dev
    :alt: Code Quality
    :target: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/?branch=dev

.. |code_intelligence| image:: https://scrutinizer-ci.com/g/boromir674/topic-modeling-toolkit/badges/code-intelligence.svg?b=dev
    :alt: Code Intelligence
    :target: https://scrutinizer-ci.com/code-intelligence

.. |version| image:: https://img.shields.io/pypi/v/topic-modeling-toolkit.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/topic-modeling-toolkit

.. |python_versions| image:: https://img.shields.io/pypi/pyversions/topic-modeling-toolkit.svg
    :alt: Supported versions
    :target: https://pypi.org/project/topic-modeling-toolkit

