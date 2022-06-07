FROM pytorch/pytorch

WORKDIR /

RUN pip install matplotlib


COPY train.py ./train.py
COPY networks ./networks

RUN python3 train.py