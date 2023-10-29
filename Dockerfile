FROM python:3.8.12-slim-bullseye as builder


# Determine where to install poetry
ENV POETRY_HOME=/opt/poetry

# Install Poetry & generate a requirements.txt file
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python

# Generate requirements.txt from poetry format
COPY poetry.lock pyproject.toml ./
RUN "$POETRY_HOME/bin/poetry" export -f requirements.txt > requirements.txt

FROM python:3.8.12-slim-bullseye


# RUN apt-get update && \
#     apt-get install -y --no-install-recommends build-essential && \
#     pip install -U pip && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Keep the requirements.txt file from the builder image
COPY --from=builder requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# Pre emptively add the user's bin folder to PATH
ENV PATH="/root/.local/bin:$PATH"

COPY . .
# RUN pip install --no-cache-dir --user .


# WORKDIR /app

# ADD imagenet-vgg-verydeep-19.mat.tar .


# COPY requirements/dev.txt reqs.txt
# RUN pip install -r reqs.txt

# COPY . .

# RUN pip install -e .

# COPY tests tests

# ENV AA_VGG_19 /app/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat
# ENTRYPOINT [ "neural-style-transfer" ]
