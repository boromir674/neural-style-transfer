# Changelog

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).

## 2.0.0 (2024-11-2)

Start official support for Python `3.10` and `3.11`.  
BREAKING: drop official support for Python `3.8` (through poetry build).

### Changes

#### Feature
- render inactive the Stop button based subscribed algorithm running/stopped state updates
- listen to whether the algorithm training/learning loop is running or interrupted
- provide text box to configure maximum epochs for the Learning Algorithm Loop to run
- prefill file selection dialogs with test Content and Style images for quicker run
- resize Style image to match Content, with CLI flag "--auto-resize-style" or "-auto"
- equip create_algo_runner with automatic Style image resize and listening to user stop signal
- broadcast algorithm running state, True for running, False for terminated, from nst tf runner
- add stop_signal callback as nst.run kwarg and notify progress subscribers on every iteration
- allow automatic resizing of Style to Content image, with Simple Resizing algorithm
- provide the Stop Signal Termination Condition that interrupts the learning loop
- implement simple logging solution with File and Console handlers both with DEBUG level

#### Fix
- prevent out-of-memory errors and unpredictable timing of garbage collection, in some cases

#### CI
- do not trigger CI/CD Pipeline on push to 'release' branch
- migrate documentation Readthedocs CI build Job to Python 3.10 (from 3.11)
- bump PyPI Resuable workflow to v1.13.1
- migrate Legacy Pylint CI Job to python 3.10
- continuously run ruff against source code: src dir
- continuously run bandit against code and fail Job if any issue is found!
- use python 3.10 in Code Visualization CI Job
- fix Static Code Analysis (ie isort, black, pyroma) on CI
- redesign Dockerfile with base Debian/Python3.11 Slim image
- update visualize-dockerfile.py script

#### Build
- pin griffe package aparently now required to solve docs dependencies build
- remove official support for Python 3.8 and 3.9 and support 3.10 and 3.11

#### Docs
- retire README.rst, in favor of Markdown README.md

#### Refactor
- eliminate all bandit issues and clean code
- apply black on source, tests, and scripts python code
- sort imports semantically and alphabetically
- rename Algorithm Data class attribute to termination_conditions for better semantics

#### Test
- expect new class RuntimeStopSignal, integrate with termination conditions & update regressions
- prevent AA_VGG_19 env var mock from persisting in test session


## 1.1.0 (2023-11-15)

### Changes

##### feature
- include wheel into the python distribution for PyPI

##### documentation
- add Doc Pages content and use material theme
- document cicd pipeline, by visualizing the Github Actoins Workflow as a graph
- automatically create nav tree of API refs, from discovered docstrings in *.py
- update README

##### build
- default docker build stage includes vgg and demo cmd

##### ci
- call reusable workflow to handle PyPI Publish Job
- run Docker Job from reusable workflow


## 1.0.1 (2023-11-05)

**CI Docker Behaviour**
- Implement an ON/OFF switch that gurantees nothing gets docker build and push
- Implement an ON/OFF switch that (given above ON), guarantees docker build and push
- Implement a switch between Continuous Deployment and Continuous Delivery Modes

**New Multi-Stage Dockerfile**
- `docker build --target prod_install .`: Image with installed app; Python and CLI
- `docker build --target prod_ready .`: Image with installed app + baked in Pretrained VGG Image Model
- `docker build --target --target prod_demo .`: Image with installed app + baked in Pretrained VGG Image Model + data to run quick NST demo
- `docker build .`: Image with installed app + baked in Pretrained VGG Image Model + default entrypoint

**One-click, containerized NST Demo run**
`docker-compose up`

### Changes

##### fix
- prioritize env vars to find the Demo Content and Style Images

##### documentation
- document ci pipeline configuration file

##### development
- multi-stage Dockerfile, and demo-nst service in docker-compose
- define distince Docker Stages for image with or without baked-in image model vgg weights
- introduce the Ruff tool for Fast! Static Code Analysis

##### refactor
- apply all static checks

##### build
- on CI Docker Job, build for Stage prod_install (--target), since missing vgg weights

##### ci
- accept 4 Policies to define Docker BUild/Publish decision making
- on/off switch, giving option of running docker Job, in case tests Job was skipped
- static checks with isort, black, pylint, pyrom, pyflakes, mccabe, dodgy, profile-validator
- deploy (built .tar.gz and/or .whl) 'v*' tags; dev: test.pypi.org, master: pypi.org


## 1.0.1-dev (2023-10-31)

Revive CI Pipeline

### Changes

##### documentation
- update badges refs and urls
- add CI Pipeline Status Badge in README
- show Demo Content Image + Style Image = Generated Image

##### ci
- Upload Code Coverage Data to Codecov.io, resulted from Test Suite runs


## 1.0.0 (2023-10-29)

- Prototype GUI Client
- over 90% code coverage
- heavily document what is going on in the code

### Changes

##### feature
- initialize same Stochastic Process on subsequent processes
- interactive GUI with Live Update of Gen Image
- nst algo - broadcast weighted costs
- add cli cmd to quickly demo algorithm on 300 x 225 Content & Style images

##### test
- running NST on indentical input (Content/Style) yields same Generated Image
- unit test the Layer bridging the backend code and the Demo CLI cmd

##### development
- devcontainer and docker-compose with tensorboard service
- install tree cli tool, inside devcontainer

##### refactor
- breakdown perform_nst method into smaller ones
- remove the 'utils ' local package and use the software-patterns' package from pypi

##### build
- migrate from setuptools to poetry build

##### ci
- add Github Actions CI Pipeline

##### demo
- on gui start up, (load and) select Demo Content/Style Images, & render UI accordingly


## 0.6.1 (2021-12-01)

### Changes

#### test
- test algorithm & conditionally mock production pretrained model

#### documentation
- document how to use the docker image hosted on docker hub

#### ci
- run regression test on ci server
