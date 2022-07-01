FROM ubuntu:20.04

RUN  apt-get update
RUN apt install wget -y
RUN apt install curl -y
RUN apt install unzip -y

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH} 

RUN conda update -y conda
RUN conda list

# create directory for notebooks
RUN mkdir /visium_task
WORKDIR /visium_task

COPY app.py /visium_task/app.py

COPY utils/ /visium_task/utils

COPY environment.yml /visium_task/environment.yml
COPY Report_SER.ipynb /visium_task/Report_SER.ipynb

RUN curl http://emodb.bilderbar.info/download/download.zip -O
RUN unzip download.zip -d ./dataset/


RUN conda update -n base -c defaults conda
RUN conda env create -f environment.yml

EXPOSE 8888
ENTRYPOINT ["conda", "run", "-n", "ml", "jupyter", "notebook","--no-browser","--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# ENTRYPOINT ["jupyter", "notebook", "--no-browser","--ip=0.0.0.0","--NotebookApp.token=''","--NotebookApp.password=''","--allow-root"]
