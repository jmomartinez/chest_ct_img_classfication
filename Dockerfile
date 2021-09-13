# DOESN'T RUN
# Tensorflow & Python versions problem
FROM python:3.8-alpine

RUN apt-get update && \
	apt-get install -y gcc make apt-transport-https ca-certificates build-essential

RUN python3 -V
RUN pip3 -V
RUN pwd

WORKDIR /src
COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

COPY src/ .
# RUN ls -la /src/*
CMD ["python3","-u", "/src/main.py"]
