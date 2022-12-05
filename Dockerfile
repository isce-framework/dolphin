FROM mambaorg/micromamba:1.1.0

# RUN apt-get update \
#     && apt-get -y upgrade --only-upgrade systemd openssl cryptsetup \
#     && apt-get install -y git

# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install --yes -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes


SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

# Install conda packages
COPY --chown=$MAMBA_USER:$MAMBA_USER env-gpu.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes


# https://github.com/mamba-org/micromamba-docker#use-the-shell-form-of-run-with-micromamba
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# TODO: workdir?
# TODO: do we care about which USER we are?
# Install dolphin from the pyproject.toml
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
RUN python -m pip install .

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "dolphin"]
CMD ["--help"]
