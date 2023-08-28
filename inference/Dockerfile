FROM ml_base

WORKDIR /app/

ENV PYTHONPATH=/app/

COPY ./inference/ /app/

RUN chmod +x wait-for-it.sh

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["./wait-for-it.sh", "postgres:5432", "--", \
    "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]