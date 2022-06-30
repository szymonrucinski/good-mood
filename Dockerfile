FROM ubuntu:latest

RUN  apt-get update
RUN apt install wget -y
RUN apt install curl -y
RUN apt install unzip -y

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
# -p path
# -f force
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH} 

#RUN source /root/.bashrc
#RUN source /root/.bash_profile

RUN conda update -y conda
RUN conda list

# create directory for notebooks
RUN mkdir /visium_task
WORKDIR /visium_task

COPY app.py /visium_task/app.py

COPY utils/ /visium_task/utils
# COPY custom_start.sh /visium_task/custom_start.sh
# RUN chmod 777 /visium_task/custom_start.sh

COPY hist.yml /visium_task/hist.yml
COPY Report_SER.ipynb /visium_task/Report_SER.ipynb

RUN curl http://emodb.bilderbar.info/download/download.zip -O
RUN unzip download.zip -d ./test_dataset/


RUN conda update conda
RUN conda env create -f hist.yml
RUN conda init
RUN activate ml
RUN conda install ipykernel
# SHELL ["conda","run","-n","ml","/bin/bash","-c"]
RUN python -m ipykernel install --name ml --display-name "ml"

# RUN conda activate ml

# RUN conda install pip
# RUN conda install libgcc
# RUN conda install -c anaconda notebook
# RUNconda install -c pytorch torchvision

# RUN pip install -r requirements.txt
# cleanup
# RUN rm Miniconda3-latest-Linux-x86_64.sh
SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]
EXPOSE 8888
ENTRYPOINT ["jupyter", "notebook", "--no-browser","--ip=0.0.0.0","--NotebookApp.token=''","--NotebookApp.password=''","--allow-root"]

# start the jupyter notebook in server mode

# CMD bash -C '/visium_task/custom_start.sh';'bash'
