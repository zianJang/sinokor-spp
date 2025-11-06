FROM ubuntu:noble
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /uvx /usr/local/bin/

# Set ARGs for non-interactive build
ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Etc/UTC

RUN <<EOT
    # Install APT packages.
    apt-get update
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        locales \
        unzip \
        vim \
        wget \
        zsh

    # Clean up APT caches to reduce image size.
    apt-get clean
    apt-get autoremove -y
    rm -rf /var/lib/apt/lists/*

    # Set up the locale.
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
    locale-gen
EOT

# Set environment variables for locale.
ENV LANG="en_US.UTF-8" \
    LANGUAGE="en_US:en" \
    LC_ALL="en_US.UTF-8"

WORKDIR /root
