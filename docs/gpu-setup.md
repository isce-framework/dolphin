# GPU setup

If you have access to a GPU with CUDA support, you can gain a considerable processing speedup with `dolphin`.
We use both [Numba](https://github.com/numba/numba/) and [JAX](https://jax.readthedocs.io/), which each have slightly different setups:

- Numba instructions: https://numba.readthedocs.io/en/stable/cuda/overview.html#software
- JAX instructions: https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu

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

For JAX, we would run

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Finally, we can also have extra GPU memory tracking by installing
```bash
mamba install pynvml
```

## JAX GPU notes

For shared environments, one important note is that JAX will pre-allocate 75% of the GPU's memory at the start of the program to avoid memory fragmentation. See the [JAX page](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html#gpu-memory-allocation) for configuration options to avoid this.

For further optimizations and tuning your specific GPU, see the [profiling notes](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#xla-performance-flags)
