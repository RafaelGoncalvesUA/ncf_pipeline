FROM python:3.9
WORKDIR /app
# COPY MODEL_PATH /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py .
CMD ["python", "serve.py"]