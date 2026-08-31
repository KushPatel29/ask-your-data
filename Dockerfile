# Containerized proof AND the deployable app.
#
# This image used to do only the first job: CMD was `pytest tests/ -v`, so the
# container ran the offline suite and exited. That is a fine CI artifact and a
# useless web service, which is exactly what Render made of it - four deploys in
# a row reported "No open ports detected" and then failed, because nothing was
# ever listening. The image had no way to serve the app it proves.
#
# So the default command now serves Streamlit, and CI overrides it to run the
# suite (`docker run --rm ask-your-data pytest tests/ -v`). Both jobs still
# happen; only the default changed.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN groupadd --system app && useradd --system --gid app --create-home app
COPY --chown=app:app . .

# Measured, not guessed: baseline 18 MB, + DuckDB warehouse 185 MB, + the Chroma
# schema index 275 MB. Streamlit's own overhead sits on top of that. Retrieval
# is warmed once at startup so the first visitor does not pay model load time;
# production sizing must leave headroom above the measured working set.
#
# The suite is NOT run here. Building the warehouse once per test module and then
# adding the embedding model is what pushed the old image past 512 MB; that work
# belongs on a CI runner with real memory, not on the box serving the demo.
ENV PORT=10000 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 10000

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','10000')+'/_stcore/health', timeout=3)"

# Shell form so $PORT expands - Render assigns the port and the service is marked
# unhealthy if nothing binds it.
CMD streamlit run app/streamlit_app.py \
      --server.port=$PORT \
      --server.address=0.0.0.0 \
      --server.headless=true
