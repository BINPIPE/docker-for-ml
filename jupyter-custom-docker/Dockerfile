FROM jupyter/scipy-notebook
MAINTAINER Prasanjit Singh // www.binpipe.org

ARG SOME_ENV_VAR
ENV SOME_ENV_VAR=${SOME_ENV_VAR}

RUN pip install awscli --upgrade --user
RUN echo 'export PATH=~/.local/bin:$PATHn' &gt;&gt; $HOME/.bashrc

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN rm ./requirements.txt
