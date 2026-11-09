FROM python:3.12-slim

RUN pip install --no-cache-dir \
    google-colab-cli==0.7.5 \
    fastapi uvicorn python-multipart

WORKDIR /app
COPY panel.py ui.html ./

ENV HOME=/data \
    PYTHONUNBUFFERED=1

RUN mkdir -p /data && chmod 777 /data
VOLUME /data

EXPOSE 8080
CMD ["uvicorn", "panel:app", "--host", "0.0.0.0", "--port", "8080"]
