FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If source data is present in the build context, precompute app artifacts at image build time.
RUN if [ -f "data/LM_CS_slim.csv.gz" ]; then \
            python precompute_lm_cost_all.py --input data/LM_CS_slim.csv.gz --outdir data; \
        fi

EXPOSE 8501

# App Service provides PORT; default 8501 keeps local container runs simple.
CMD ["sh", "-c", "streamlit run lm_cost_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]