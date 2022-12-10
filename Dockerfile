FROM mambaorg/micromamba:1.1.0

# Install conda packages
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements-conda.txt /tmp/requirements.txt
RUN micromamba install --yes --channel conda-forge -n base -f /tmp/requirements.txt && \
    micromamba clean --all --yes


SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

# https://github.com/mamba-org/micromamba-docker#use-the-shell-form-of-run-with-micromamba
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# TODO: workdir?
# TODO: do we care about which USER we are?
# Install dolphin from the pyproject.toml
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
# --no-deps because they are installed with conda
RUN python -m pip install --no-deps .

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "dolphin"]
CMD ["--help"]
