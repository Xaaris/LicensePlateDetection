FROM tensorflow/tensorflow:1.8.0-gpu-py3

RUN apt update && apt install -y libsm6 libxext6 libfontconfig1 libxrender1 ffmpeg

RUN pip --no-cache-dir install numpy
RUN pip --no-cache-dir install moviepy
RUN pip --no-cache-dir install opencv-python

COPY . /app

# don't run as root
RUN groupadd -g 1000 appuser && useradd -r -u 1000 -g appuser appuser
USER appuser

WORKDIR /app

ENTRYPOINT ["python3", "main.py"]


