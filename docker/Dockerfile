FROM mambaorg/micromamba:1.1.0@sha256:53697a4bae2d9a4407463b8f314d370e201c58f6b5e5ff7c1f24699ed7e37a41

# Install conda packages
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/specfile.txt /tmp/specfile.txt
RUN micromamba install --yes --channel conda-forge -n base -f /tmp/specfile.txt && \
    micromamba clean --all --yes



# Install dolphin from the pyproject.toml
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# activte, otherwise python will not be found
# https://github.com/mamba-org/micromamba-docker#running-commands-in-dockerfile-within-the-conda-environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# --no-deps because they are installed with conda
RUN python -m pip install --no-deps .

# SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "-m", "dolphin"]
# CMD ["--help"]
