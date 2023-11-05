# Changelog


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
