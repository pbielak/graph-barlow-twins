FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu18.04

RUN apt-get update && \
    apt-get install -y software-properties-common wget git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update

RUN apt-get install -y --no-install-recommends \
        python3.7 \
        python3.7-dev

RUN wget https://bootstrap.pypa.io/get-pip.py &&  \
	 python3.7 get-pip.py && \
	 ln -s /usr/bin/python3.7 /usr/local/bin/python3

RUN apt-get install -y zsh && \
    chsh -s /bin/zsh && \
    zsh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"

ADD . /app
WORKDIR /app

RUN apt-get install -y $(awk '{print $1}' docker/apt.packages)

RUN pip install -r ./requirements.txt

EXPOSE 8888

CMD exec /bin/zsh -c "trap : TERM INT; sleep infinity & wait"
