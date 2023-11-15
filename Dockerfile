### Start Stage: Use Python 3.8
FROM python:3.8.12-slim-bullseye as base

FROM base as source

WORKDIR /app

# Copy Source Code
# COPY . .

# TODO improve below (chached wheels could help)
COPY src src
COPY CHANGELOG.md .
COPY pyproject.toml .
COPY poetry.lock .
COPY LICENSE .
COPY README.rst .
# COPY Pretrained_Model_LICENSE.txt .


# BRANCH 1

### Prod Build Wheels ###

FROM source AS prod

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry-core

# Build Wheels for package and its python dependencies
RUN pip wheel --wheel-dir /app/dist /app


FROM base AS prod_install

# this Stage does not need Source Code

WORKDIR /app

# we extract the wheels built in 'prod'
COPY --from=prod /app/dist dist

# Install wheels for python package and its deps
# in user site-packages (ie /root/.local/lib/python3.8/site-packages)
RUN pip install --no-cache-dir --user ./dist/*.whl

# all code-wise has been successfully built (wheels) and installed

# Distribution installed (prod wheels)

# Optionaly, add the CLI executable to the PATH
# to make the `nst` CLI available in the image
ENV PATH="/root/.local/bin:$PATH"

## This is a Builder kind of Stage
## Consider using the docker feature on-build

# build: docker build --target prod_install -t nst_cli .
# Usable as: docker run -it --rm nst_cli --entrypoint nst

# EG: docker run -it --rm nst_cli nst --help

CMD [ "nst" ]

### END of Prod Build Installation ###



### Stage: Bake Model Weights into Image ###

FROM prod_install AS prod_ready

ENV PATH="/root/.local/bin:$PATH"
# Now the App CLI is available as `nst`


## USAGE: as a dev you want to create images with baked in weights

# VGG weights must be in the 'build context', when running `docker build`

# Use Case 1
# the file with vgg weights are in (relative) path
# ./pretrained_model_bundle/imagenet-vgg-verydeep-19.mat

# docker build -t nst-vgg --target prod_ready .

# Use Case 2
# the file with vgg weights are in (relative) path
# ./deep-neural-nets/imagenet-vgg-verydeep-19.mat

# docker build --build-arg REPO_DIR="${PWD}/deep-neural-nets" -t nst-vgg --target prod_ready .

# the file with vgg weights are in (absolute) path
# /my_filesystem/data/repos/neural-style-transfer/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat


# USE examples:

#  docker build --target prod_ready -t nst-vgg \
#      --build-arg BUILD_TIME_VGG="/host_specific/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat" \
#      .

#  docker build --target prod_ready -t nst-vgg \
#      --build-arg REPO_DIR="/host_specific" \
#      .

#  docker build --target prod_ready -t nst-vgg \
#      --build-arg REPO_DIR="/data/repos/neural-style-transfer" \
#      .

#  docker build --target prod_ready -t nst-vgg \
#      --build-arg BUILD_TIME_VGG="/data/repos/neural-style-transfer/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat" \
#      .

# POC for persisting a variable as env var available in the image after build
# ARG CONT_IMG_VER
# ENV CONT_IMG_VER=${CONT_IMG_VER:-v1.0.0}
# END of POC

## Build Time Args ##
# not available at run time, like ENV vars

ARG VGG_NAME="imagenet-vgg-verydeep-19.mat"

# Overide most probably
ARG DEFAULT_VGG_DIR_NAME="pretrained_model_bundle"

# Overide most probably
# ARG DEFAULT_REPO_DIR="/data/repos/neural-style-transfer"
# ARG IMAGE_MODEL="${DEFAULT_REPO_DIR}/${DEFAULT_VGG_DIR_NAME}/${VGG_NAME}"

# IMAGE_MODEL in 'build context'; relative path to the context of the build
# ie `docker build -t im .` means context is $PWD
ARG IMAGE_MODEL="${DEFAULT_VGG_DIR_NAME}/${VGG_NAME}"

WORKDIR /app

# # Copy a file from the host system to the image
# # preserving same folder structure. It is optional, but for purposes of clarity
COPY ${IMAGE_MODEL} "${DEFAULT_VGG_DIR_NAME}/"

# ie if on host is pretrained_model_bundle/imagenet-vgg-verydeep-19.mat
# then in image it will be /app/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat


# # Make Image Model's VGG wieghts avalilable to NST code/app
ENV AA_VGG_19="/app/${DEFAULT_VGG_DIR_NAME}/${VGG_NAME}"

# CMD that verifies file exist in image with default build args:
CMD ["python", "-c", "import os; print(os.path.exists(os.environ['AA_VGG_19']))"]


### Stage: Bake into image the Demo Content and Style Images

FROM prod_ready AS prod_demo


ARG REPO_DEMO_IMAGES_LOCATION="tests/data"

ARG DEMO_CONTENT_IMAGE="${REPO_DEMO_IMAGES_LOCATION}/canoe_water_w300-h225.jpg"
ARG DEMO_STYLE_IMAGE="${REPO_DEMO_IMAGES_LOCATION}/blue-red_w300-h225.jpg"

WORKDIR /app

COPY ${DEMO_CONTENT_IMAGE} "${REPO_DEMO_IMAGES_LOCATION}/"
COPY ${DEMO_STYLE_IMAGE} "${REPO_DEMO_IMAGES_LOCATION}/"
# ie if on host is tests/data/blue-red_w300-h225.jpg
# then in image it will be /app/tests/data/blue-red_w300-h225.jpg

# Indicate that these are valid Content and Style Images for Demo
# Show nst where to find the Demo Content and Style Images
ENV CONTENT_IMAGE_DEMO="/app/${DEMO_CONTENT_IMAGE}"
ENV STYLE_IMAGE_DEMO="/app/${DEMO_STYLE_IMAGE}"


# Define default command, that when run, a python process is spawned, which
# runs the NST Algorithm on the Demo Content and Style Images for a few iterations
CMD ["nst", "demo"]


FROM prod_ready as default

CMD [ "nst" ]

### Stage: Default Target (for Production)
# Just to allow `docker buil` use this as target if not specified

FROM prod_demo as default_with_demo

# Define ENTRYPOINT, so that this is the default 
# runs the NST Algorithm on the Demo Content and Style Images for a few iterations

ENTRYPOINT [ "nst" ]
# overwrite: docker run --entrypoint /bin/bash -it --rm nst-prod-runtime




## Notes

# if there is aneed that the image model weight get into the image at build-time
# then we think of 2 approaches:

# OPT 1
# A Stage here, where the image is copied from host and then we can build 2 versions
# for dockerhub, an image with and image without pre-baked starting model weights

# OPT 2
# or we supply a docker build time argument that lets the user set the path to
# the model weights, 




# get rid of Stage 1 and keep only requirements.txt from it
# Get rid of prev image and start fresh


# Probably will need tools for building tensorflow, numpy, scipy
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends build-essential && \
#     pip install -U pip && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Research:
# Do we have both tar.gz and wheel(s)?
# where are they?
# ideally, we want to have available here all prebuilt wheels
# required for our artificial_artwork python package and all its
# python dependencies, so that we install as many wheels as possible
# but why, to avoid
# WORKDIR /app

# ADD imagenet-vgg-verydeep-19.mat.tar .

# COPY requirements/dev.txt reqs.txt
# RUN pip install -r reqs.txt

# COPY . .

# RUN pip install -e .

# COPY tests tests

# ENTRYPOINT [ "neural-style-transfer" ]

