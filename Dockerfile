FROM python:3.10.15-slim-bullseye

# Install dependencies
RUN apt update; \
    apt install -y git

# Upgrade pip
RUN pip install --upgrade pip

# Install avnirpy
RUN pip install git+https://github.com/llgneuroresearch/avnirpy.git

# Set entrypoint
ENTRYPOINT [""]