FROM "intelaipg/intel-optimized-tensorflow"
RUN mkdir -p /tensorflow/models
RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y git python-pip protobuf-compiler python-pil python-lxml
RUN pip install --upgrade pip
RUN pip install intel-tensorflow matplotlib flask
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models
WORKDIR /tensorflow/models/research
RUN wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip -O protoc.zip
RUN unzip protoc.zip
RUN ./bin/protoc object_detection/protos/*.proto --python_out=.
RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
EXPOSE 8888
EXPOSE 5000
WORKDIR /tensorflow/models/research/object_detection
CMD ["python", "flask_app.py"]