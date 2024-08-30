# This is the image we are going add micromaba to:
ARG BASE=ubuntu:22.04@sha256:7a57c69fe1e9d5b97c5fe649849e79f2cfc3bf11d10bbd5218b4eb61716aebe6

# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.1.0@sha256:53697a4bae2d9a4407463b8f314d370e201c58f6b5e5ff7c1f24699ed7e37a41 AS micromamba

FROM $BASE

# Needed to redefine the arg so we can use it later
ARG BASE

# Label image following opencontainers image-spec annotations recommendation:
# https://github.com/opencontainers/image-spec/blob/main/annotations.md
LABEL org.opencontainers.image.description="Container for InSAR PS/DS phase analysis using dolphin"
LABEL org.opencontainers.image.authors="Scott Staniewicz <scott.j.staniewicz@jpl.nasa.gov>"
LABEL org.opencontainers.image.url="https://github.com/isce-framework/dolphin"
LABEL org.opencontainers.image.source="https://github.com/isce-framework/dolphin"
LABEL org.opencontainers.image.documentation="https://dolphin-insar.readthedocs.io/en/latest/"
LABEL org.opencontainers.image.licenses="BSD-3-Clause OR Apache-2.0"

# Install CA certificates if the base image has apt-get, otherwise we can't install conda packages
RUN if command -v apt-get > /dev/null; then \
    apt-get update && \
    apt-get install -y ca-certificates wget && \
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
    # cxx-compiler for whirlwind:
    micromamba install --yes --channel conda-forge -n base cxx-compiler && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Activate, otherwise python will not be found
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# For now, manually install tophu/snaphu from git/pip
RUN python -m pip install --no-deps \
    "snaphu>=0.4" \
    git+https://github.com/isce-framework/tophu@66df96fc4645f6977421336ed96c553833216c07 \
    git+https://github.com/isce-framework/spurt@main \
    git+https://github.com/isce-framework/whirlwind@40defb38d2d6deca2819934788ebbc57e418e32d

# --no-deps because they are installed with conda
RUN python -m pip install --no-deps .

# Run the entrypoint from the /work directory
# This means if they mount a volume to /work, we won't be mixing in the repository code
WORKDIR /work

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["dolphin", "--help"]
