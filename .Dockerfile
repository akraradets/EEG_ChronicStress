# FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/Bangkok"

# like CD command in terminal. it will create directory if path is not existed
WORKDIR /root/projects
RUN apt update && apt upgrade -y
# Set timezone
RUN apt install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Usual terminal commands for installing environment
RUN apt install python3 python3-pip -y
RUN apt install git -y
# I will use `pipenv` to dynamically controll my environment
# If you want to use `pip install`, just remove `pipenv` and continue with `pip install`
RUN pip install pipenv

CMD tail -f /dev/null