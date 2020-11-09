FROM jupyter/scipy-notebook
MAINTAINER Prasanjit Singh // www.binpipe.org

RUN pip install joblib

RUN mkdir model
ENV MODEL_DIR=/home/jovyan/model
ENV MODEL_FILE=clf.joblib
ENV METADATA_FILE=metadata.json

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py
