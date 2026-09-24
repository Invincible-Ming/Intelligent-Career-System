FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
RUN apt-get update && apt-get install -y --no-install-recommends iptables util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 mcp && useradd --uid 10001 --gid 10001 --no-create-home mcp
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY policy.py db_guard.py server.py broker.py entrypoint.py ./
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 10001:10001
CMD ["python", "-u", "server.py", "search"]
