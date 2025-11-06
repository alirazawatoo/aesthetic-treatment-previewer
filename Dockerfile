FROM python:3.12-slim

# requirements.txt (default) runs the core app. requirements-dev.txt adds the
# test tools; requirements-landmarks.txt swaps in the MediaPipe stack, whose
# opencv-contrib wheel needs the system GL and GLib libraries.
ARG REQUIREMENTS=requirements.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements*.txt ./
RUN if grep -q mediapipe "$REQUIREMENTS"; then \
        apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*; \
    fi \
    && pip install -r "$REQUIREMENTS"

COPY . .
RUN useradd --create-home --uid 10001 preview && chown -R preview /app
USER preview

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
