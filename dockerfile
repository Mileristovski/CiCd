FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip && \
    python -m pip install -U scikit-learn 
RUN python -m pip install pandas Flask

COPY . /app

CMD ["python", "app.py"]
