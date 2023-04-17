ARG BASE=opera-adt/dolphin:latest

# bring in the micromamba image so we can copy files from it
FROM $BASE

ARG MAMBA_USER=mamba

# Install extra conda packages for testing
# https://github.com/mamba-org/micromamba-docker#quick-start
COPY --chown=$MAMBA_USER:$MAMBA_USER tests/requirements.txt /tmp/test-requirements.txt
RUN micromamba install --yes --channel conda-forge -n base -f /tmp/test-requirements.txt && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . .
