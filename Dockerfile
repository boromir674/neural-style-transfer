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
# Install the Build Backend for the pip Frontend
# poetry-core 1.8.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry-core

# tools for building tensorflow, numpy, scipy
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends build-essential && \
#     pip install -U pip && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Build Wheels for package and its python dependencies
RUN pip wheel --wheel-dir /app/dist /app

# Keep only wheels from previous Stage and install ###
FROM base AS prod_install

WORKDIR /app

COPY --from=prod /app/dist dist

RUN pip install --no-cache-dir --user ./dist/*.whl

# Add now the user's bin folder to PATH
ENV PATH="/root/.local/bin:$PATH"

# Now the App CLI is available as `nst`
# all code-wise has been successfully built (wheels) and installed

# END of IMAGE v1 (prod)


### Stage: Bake Model Weights into Image ###

# Responsible for baking into the image the pre-trained model weights

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


FROM prod AS prod_ready

## USAGE: as a dev you want to create images with baked in weights

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

# POC for persisting a variable as env var available in the image after build
# ARG CONT_IMG_VER
# ENV CONT_IMG_VER=${CONT_IMG_VER:-v1.0.0}
# END of POC

## Build Time Args ##

ARG VGG_NAME="imagenet-vgg-verydeep-19.mat"

# Overide most probably
ARG DEFAULT_VGG_DIR_NAME="pretrained_model_bundle"
# Overide most probably
ARG DEFAULT_REPO_DIR="/data/repos/neural-style-transfer"


# IMAGE_MODEL references the Host's Filesystem, with default value
ARG IMAGE_MODEL="${DEFAULT_REPO_DIR}/${DEFAULT_VGG_DIR_NAME}/${VGG_NAME}"

# ARG IMAGE_MODEL="/data/repos/neural-style-transfer/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat"

# Print variable values for debugging
RUN echo "IMAGE_MODEL: ${IMAGE_MODEL}" > aa
RUN echo "DEFAULT_VGG_DIR_NAME: ${DEFAULT_VGG_DIR_NAME}" >> aa
RUN echo "DEFAULT_REPO_DIR: ${DEFAULT_REPO_DIR}" >> aa
RUN echo "VGG_NAME: ${VGG_NAME}" >> aa


# WORKDIR $MY_WORKDIR
# WORKDIR /app

# # Copy a file from the host system to the image
# # preserving same folder structure. It is optional, but for purposes of clarity
# COPY ${IMAGE_MODEL} ${DEFAULT_VGG_DIR_NAME}

# # Make Image Model's VGG wieghts avalilable to NST code/app
# ENV AA_VGG_19=/app/${DEFAULT_VGG_DIR_NAME}/VGG_NAME


# ### Stage: Bake Model Weights into Image ###
# FROM prod_ready AS prod_ready_with_demo

# WORKDIR /app

# # Bake into image the Demo Content and Style Images
# COPY tests/data/blue-red_w300-h225.jpg /app/tests/data/blue-red_w300-h225.jpg
# COPY tests/data/canoe_water_w300-h225.jpg /app/tests/data/canoe_water_w300-h225.jpg

# # Define default command, that when run, a python process is spawned, which
# # runs the NST Algorithm on the Demo Content and Style Images for a few iterations

# CMD ["nst", "demo"]




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

# ENV AA_VGG_19 /app/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat
# ENTRYPOINT [ "neural-style-transfer" ]
