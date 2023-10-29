# Changelog


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
