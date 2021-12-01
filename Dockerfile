FROM python:3.8-slim

WORKDIR /app

ADD imagenet-vgg-verydeep-19.mat.tar .


COPY requirements/dev.txt reqs.txt
RUN pip install -r reqs.txt

COPY . .

RUN pip install -e .

COPY tests tests

ENV AA_VGG_19 /app/pretrained_model_bundle/imagenet-vgg-verydeep-19.mat
ENTRYPOINT [ "neural-style-transfer" ]
