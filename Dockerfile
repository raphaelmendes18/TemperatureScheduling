FROM python:3.8.2

RUN pip install --upgrade pip
RUN pip install numpy==1.17.4 
RUN python -m pip install "dask[delayed]" 
RUN mkdir /usr/src/scheduler

COPY src/* /usr/src/scheduler/

WORKDIR /usr/src/scheduler
