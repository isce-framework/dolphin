# GPU setup

If you have access to a GPU with CUDA support, you can gain a considerable processing speedup with `dolphin`.
We use both [Numba](https://github.com/numba/numba/) and [Jax](jax.readthedocs.io/), which each have slightly different setups:

- Numba instructions: https://numba.readthedocs.io/en/stable/cuda/overview.html#software
- Jax instructions: https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu

Both of these require you to install software *which matches your version of CUDA*.
You can check which CUDA version is installed with `nvidia-smi`:

```bash
$ nvidia-smi | grep -i version
| NVIDIA-SMI 510.39.01    Driver Version: 510.39.01    CUDA Version: 11.6     |
```

We see that there is version 11.6, so for Numba, we would install

```bash
mamba install cudatoolkit=11.6
```

For Jax, we would run

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Finally, we can also have extra GPU memory tracking by installing
```bash
mamba install pynvml
```
