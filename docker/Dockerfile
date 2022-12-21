# bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.1.0@sha256:53697a4bae2d9a4407463b8f314d370e201c58f6b5e5ff7c1f24699ed7e37a41 as micromamba

# This is the image we are going add micromaba to:
# See https://github.com/oracle/container-images or https://hub.docker.com/_/oraclelinux
FROM oraclelinux:8-slim

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
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
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

# activte, otherwise python will not be found
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# --no-deps because they are installed with conda
RUN python -m pip install --no-deps .


ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m", "dolphin"]
CMD ["--help"]
