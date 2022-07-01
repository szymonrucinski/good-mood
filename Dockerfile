FROM ubuntu:20.04

RUN  apt-get update
RUN apt install wget -y
# RUN apt install curl -y
RUN apt install unzip -y

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH} 

RUN conda update -y conda
RUN conda list
RUN mkdir /visium_task

WORKDIR /visium_task

COPY app.py /visium_task/app.py
COPY utils/ /visium_task/utils
RUN wget https://filedn.eu/lJe8HQehDK0jkgvBcE4bDl8/model.pt


COPY environment.yml /visium_task/environment.yml
COPY Report_SER.ipynb /visium_task/Report_SER.ipynb

RUN wget http://emodb.bilderbar.info/download/download.zip
RUN unzip download.zip -d ./dataset/


RUN conda update -n base -c defaults conda
RUN conda env create -f environment.yml


COPY entry_point.sh /visium_task/entry_point.sh
RUN ["chmod", "+x", "/visium_task/entry_point.sh"]
ENTRYPOINT ["/visium_task/entry_point.sh"]