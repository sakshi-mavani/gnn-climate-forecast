FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    torch

EXPOSE 8000

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]