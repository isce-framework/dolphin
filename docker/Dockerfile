# This is the image we are going add micromaba to:
# See https://github.com/oracle/container-images or https://hub.docker.com/_/oraclelinux
ARG BASE=oraclelinux:8-slim@sha256:5da605ca1fe5a75824cdca00b3a5f39907a2057c0cc0c34c48feb33b4915a3e6

# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.1.0@sha256:53697a4bae2d9a4407463b8f314d370e201c58f6b5e5ff7c1f24699ed7e37a41 AS micromamba

FROM $BASE

# Needed to redefine the arg so we can use it later
ARG BASE

# Label image following opencontainers image-spec annotations recommendation:
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.description="Container for InSAR PS/DS phase analysis using dolphin"
LABEL org.opencontainers.image.authors="Scott Staniewicz <scott.j.staniewicz@jpl.nasa.gov>"
LABEL org.opencontainers.image.url="https://github.com/opera-adt/dolphin"
LABEL org.opencontainers.image.source="https://github.com/opera-adt/dolphin"
LABEL org.opencontainers.image.documentation="https://dolphin-insar.readthedocs.io/en/latest/"
LABEL org.opencontainers.image.licenses="BSD-3-Clause OR Apache-2.0"

# Install CA certificates if the base image is Ubuntu, otherwise we can't install conda packages
RUN if [ "$(echo $BASE | awk -F: '{print $1}')" = "ubuntu" ]; then \
    apt-get update && \
    apt-get install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*; \
    fi

ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
# The `RUN true` is a weird hack to prevent "layer does not exist" errors
# apparently arising from multi-stage builds? https://stackoverflow.com/a/62409523/4174466
RUN true
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
RUN true
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
RUN true
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh


# Location to install dolphin from the pyproject.toml
WORKDIR /dolphin
# Allow us to create new files in the workdir when pip installing
RUN chown $MAMBA_USER:$MAMBA_USER /dolphin

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]


# Install conda packages
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/specfile.txt /tmp/specfile.txt
RUN micromamba install --yes --channel conda-forge -n base -f /tmp/specfile.txt && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Activate, otherwise python will not be found
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# --no-deps because they are installed with conda
RUN python -m pip install --no-deps .

# Run the entrypoint from the /work directory
# This means if they mount a volume to /work, we won't be mixing in the repository code
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["dolphin", "--help"]
