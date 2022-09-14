# FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
FROM ubuntu:22.04

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
# RUN pip install pipenv

RUN pip install ipykernel
RUN pip install numpy
RUN pip install pandas
RUN pip install matplotlib
RUN pip install sklearn
RUN pip install mne mne-features

# # RUN pip install appdirs==1.4.4
# # RUN pip install asttokens==2.0.5
# # RUN pip install backcall==0.2.0
# # RUN pip install certifi==2022.6.15
# # RUN pip install charset-normalizer==2.0.12
# # RUN pip install cycler==0.11.0
# # RUN pip install debugpy==1.6.0
# # RUN pip install decorator==5.1.1
# # RUN pip install entrypoints==0.4
# # RUN pip install executing==0.8.3
# # RUN pip install fonttools==4.33.3
# # RUN pip install idna==3.3
# RUN pip install ipykernel==6.15.0
# # RUN pip install ipython==8.4.0
# # RUN pip install jedi==0.18.1
# # RUN pip install Jinja2==3.1.2
# # RUN pip install joblib==1.1.0
# # RUN pip install jupyter-client==7.3.4
# # RUN pip install jupyter-core==4.10.0
# # RUN pip install kiwisolver==1.4.3
# # RUN pip install llvmlite==0.38.1
# # RUN pip install MarkupSafe==2.1.1
# RUN pip install matplotlib==3.5.2
# # RUN pip install matplotlib-inline==0.1.3
# RUN pip install mne==1.0.3
# RUN pip install mne-features==0.2
# # RUN pip install nest-asyncio==1.5.5
# # RUN pip install numba==0.55.2
# RUN pip install numpy==1.22.4
# # RUN pip install packaging==21.3
# RUN pip install pandas==1.4.3
# # RUN pip install parso==0.8.3
# # RUN pip install pexpect==4.8.0
# # RUN pip install pickleshare==0.7.5
# RUN pip install Pillow==9.1.1
# # RUN pip install pip==22.0.4
# # RUN pip install pooch==1.6.0
# # RUN pip install prompt-toolkit==3.0.29
# RUN pip install psutil==5.9.1
# # RUN pip install ptyprocess==0.7.0
# # RUN pip install pure-eval==0.2.2
# # RUN pip install Pygments==2.12.0
# # RUN pip install pyparsing==3.0.9
# # RUN pip install python-dateutil==2.8.2
# # RUN pip install pytz==2022.1
# # RUN pip install PyWavelets==1.3.0
# # RUN pip install pyzmq==23.2.0
# RUN pip install requests==2.28.0
# RUN pip install scikit-learn==1.1.1
# RUN pip install scipy==1.8.1
# # RUN pip install setuptools==62.6.0
# # RUN pip install six==1.16.0
# # RUN pip install sklearn==0.0
# # RUN pip install stack-data==0.3.0
# # RUN pip install threadpoolctl==3.1.0
# # RUN pip install tornado==6.1
# # RUN pip install tqdm==4.64.0
# # RUN pip install traitlets==5.3.0
# # RUN pip install urllib3==1.26.9
# # RUN pip install wcwidth==0.2.5
# # RUN pip install wheel==0.37.1

CMD tail -f /dev/null