
## Build-time Python Interpreter Version (--build-arg PY_VERSION=3.11.10)
ARG PY_VERSION=3.11.10

FROM python:${PY_VERSION}-slim-bullseye as base
# FROM python:${PY_VERSION}-alpine3.20 as base

## Provide Poetry
FROM base as poetry
COPY poetry.lock pyproject.toml ./
ENV POETRY_HOME=/opt/poetry
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python


FROM poetry AS pip_deps_prod
RUN "$POETRY_HOME/bin/poetry" export -f requirements.txt > requirements.txt


FROM scratch as source
WORKDIR /app
COPY --from=pip_deps_prod requirements.txt .
# Copy Source Code
# COPY . .
COPY src src
COPY CHANGELOG.md .
COPY pyproject.toml .
COPY poetry.lock .
COPY LICENSE .
COPY README.rst .
# COPY Pretrained_Model_LICENSE.txt .

## Provides Python Runtime and DISTRO_WHEELS folder
FROM base as base_env

# Wheels Directory for Distro and its Dependencies (aka requirements)
ENV DISTRO_WHEELS=/app/dist


### Build Wheels for Prod ###
FROM base_env AS build_wheels

# RUN apk update && \
#     apk add --no-cache build-base && \
#     pip install --upgrade pip && \
#     rm -rf /var/cache/apk/*

# Install Essential build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install -U pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Essential build-time dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry-core && \
    pip install --no-cache-dir build

WORKDIR /app
COPY --from=source /app .

# Build Wheels for package and its python dependencies
RUN pip wheel --wheel-dir "${DISTRO_WHEELS}" -r ./requirements.txt
# RUN pip wheel --wheel-dir /app/dist /app

# Build Wheels for Distro's Package
RUN python -m build --outdir "/tmp/build-wheels" && \
    mv /tmp/build-wheels/*.whl "${DISTRO_WHEELS}"

CMD [ "ls", "-l", "/app/dist" ]

### Install pre-built wheels
FROM base_env AS install
ENV DIST_DIR=dist
WORKDIR /app

# we copy the wheels built in 'build_wheels' stage
COPY --from=build_wheels ${DISTRO_WHEELS} ./${DIST_DIR}
# RUN ls -l dist

# Install wheels for python package and its deps
# in user site-packages (ie /root/.local/lib/python3.8/site-packages)
RUN pip install --no-cache-dir --user ./${DIST_DIR}/*.whl
# RUN pip install --no-cache-dir --user ${DISTRO_WHEELS}/*.whl
# Optionaly, add the CLI executable to the PATH
# to make the `nst` CLI available in the image
ENV PATH="/root/.local/bin:$PATH"

CMD [ "nst" ]


### Non-default Stage: Bake Model Weights into Image ###
FROM install AS prod_ready

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

### Stage: Default Target (for Production)
# Just to allow `docker buil` use this as target if not specified

FROM prod_demo as default_with_demo

# Define ENTRYPOINT, so that this is the default 
# runs the NST Algorithm on the Demo Content and Style Images for a few iterations

ENTRYPOINT [ "nst" ]
# overwrite: docker run --entrypoint /bin/bash -it --rm nst-prod-runtime


FROM prod_ready as default

CMD [ "nst" ]
