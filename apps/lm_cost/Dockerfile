FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# App Service provides PORT; default 8501 keeps local container runs simple.
CMD ["sh", "-c", "streamlit run lm_cost_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]