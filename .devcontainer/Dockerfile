FROM mambaorg/micromamba

COPY --from=tianon/gosu /usr/local/bin/gosu /usr/local/bin/gosu
ARG NEW_MAMBA_USER=mambauser
ARG NEW_MAMBA_USER_ID=1000
ARG NEW_MAMBA_USER_GID=1000
USER root
RUN if grep -q '^ID=alpine$' /etc/os-release; then \
      # alpine does not have usermod/groupmod
      apk add --no-cache --virtual temp-packages shadow; \
    fi && \
    usermod "--login=${NEW_MAMBA_USER}" "--home=/home/${NEW_MAMBA_USER}" \
        --move-home "-u ${NEW_MAMBA_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${NEW_MAMBA_USER}" \
        "-g ${NEW_MAMBA_USER_GID}" "${MAMBA_USER}" && \
    if grep -q '^ID=alpine$' /etc/os-release; then \
      # remove the packages that were only needed for usermod/groupmod
      apk del temp-packages; \
    fi && \
    # Update the expected value of MAMBA_USER for the
    # _entrypoint.sh consistency check.
    echo "${NEW_MAMBA_USER}" > "/etc/arg_mamba_user" && \
    :
# Update package lists and install X11-related libraries for Qt GUI support
RUN apt-get update && apt-get install -y \
    libx11-dev \
    libxext6 \
    libxrender-dev \
    libxkbcommon-x11-0 \
 && rm -rf /var/lib/apt/lists/*
ENV MAMBA_USER=$NEW_MAMBA_USER
RUN --mount=type=cache,target=/var/cache/apt apt-get update
RUN --mount=type=cache,target=/var/cache/apt apt-get install -y git
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN --mount=type=cache,mode=0777,uid=1000,gid=1000,target=/opt/conda/pkgs \
    CONDA_VERBOSITY=-2 micromamba install -n base -y -f /tmp/env.yaml
WORKDIR /home/${NEW_MAMBA_USER}/dev-container