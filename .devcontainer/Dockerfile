# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.10"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Python Configuration
ENV PYTHONUNBUFFERED 1

# Generic/Other Tools Provision/Configuration
ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# Options
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USE_MOBY="true"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/
RUN apt-get update \
    && apt-get -y install --no-install-recommends iproute2 \
    && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" "true" "true" \
    && /bin/bash /tmp/library-scripts/docker-debian.sh "true" "/var/run/docker-host.sock" "/var/run/docker.sock" "${USERNAME}" "${USE_MOBY}" \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/
    # && /bin/bash /tmp/library-scripts/kubectl-helm-debian.sh "latest" "latest" "none" \

# Application/Dev Environment Configuration
# Install HDF5 C++ Runtime files
RUN apt-get update \
    && apt-get -y install --no-install-recommends libhdf5-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# Install prerequisites to building the scipy package
# for building the scipy package we use the pip install, which for scipy==1.4
# uses the legacy build backend of setuptools
RUN apt-get update \
    && apt-get -y install --no-install-recommends gcc \
        build-essential \
        zlib1g-dev \
        wget \
        unzip \
        cmake \
        python3-dev \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*


# Install Cloud CLI's here, if needed

WORKDIR /config

# Setting the ENTRYPOINT to docker-init.sh will configure non-root access to 
# the Docker socket if "overrideCommand": false is set in devcontainer.json. 
# The script will also execute CMD if you need to alter startup behaviors.
ENTRYPOINT [ "/usr/local/share/docker-init.sh" ]
CMD [ "sleep", "infinity" ]

# Essential/Usefull Tools
# Install poetry and tox, which are essential Tools for any Python dev stack
    # && git clone git@github.com:boromir674/tree-1.8.0.git \
RUN pip3 --disable-pip-version-check install tox==3.27.1 poetry \
    && rm -rf /tmp/pip-tmp

# Install tree cli tool
RUN git clone https://github.com/boromir674/tree-1.8.0.git tree-repo
WORKDIR /config/tree-repo

RUN rm *.o && make tree && make install

WORKDIR /config

RUN rm -rf tree-repo
    # && rm -rf tree-1.8.0 \
    # && apt-get update \
    # && apt-get -y install --no-install-recommends tree \
    # && apt-get autoremove -y \
    # && apt-get clean -y \
    # && rm -rf /var/lib/apt/lists/*



# allow code navigation inside vscode, by installing the prod + test dependencies
COPY poetry.lock /tmp/pip-tmp/
COPY pyproject.toml /tmp/pip-tmp/
RUN poetry export --directory=/tmp/pip-tmp/ -f requirements.txt -o /tmp/pip-tmp/requirements-test.txt -E test \ 
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements-test.txt \
    && rm -rf /tmp/pip-tmp
