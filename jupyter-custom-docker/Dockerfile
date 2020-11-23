FROM jupyter/scipy-notebook
MAINTAINER Prasanjit Singh // www.binpipe.org

ARG SOME_ENV_VAR
ENV SOME_ENV_VAR=${SOME_ENV_VAR}

RUN pip install awscli --upgrade --user
RUN echo 'export PATH=~/.local/bin:$PATH' >> $HOME/.bashrc

COPY requirements.txt ./requirements.txt
RUN pip install --use-feature=2020-resolver -r requirements.txt
RUN rm ./requirements.txt
