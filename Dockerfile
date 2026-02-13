FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY tabpfn-v2.5-classifier-v2.5_default.ckpt /app/models/
COPY handler.py .

CMD ["python", "-u", "handler.py"]
