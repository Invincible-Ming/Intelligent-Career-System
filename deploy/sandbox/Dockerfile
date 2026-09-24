FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd --gid 10001 sandbox && useradd --uid 10001 --gid 10001 --no-create-home sandbox
WORKDIR /app
RUN pip install --no-cache-dir 'PyMuPDF>=1.24,<2.0' 'python-docx>=1.1,<2.0' 'openpyxl>=3.1,<4.0'
COPY backend/app/services/document_pipeline.py /app/document_pipeline.py
COPY deploy/sandbox/worker.py /app/worker.py
USER 10001:10001
ENTRYPOINT ["python", "-u", "/app/worker.py"]
