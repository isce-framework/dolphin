FROM opera-adt/dolphin

# Install conda packages
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/specfile-gpu.txt /tmp/specfile-gpu.txt
RUN micromamba install --yes --channel conda-forge -n base -f /tmp/specfile-gpu.txt && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . .
