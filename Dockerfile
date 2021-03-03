FROM python:3.8

EXPOSE 8080

COPY ./app /app
COPY ./test /test

RUN pip install -r /app/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]