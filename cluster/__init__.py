"""Cluster helpers: checkpoint discovery, auto-resume, run metadata, and CUDA
memory observability for running this repo on a Slurm GPU cluster.

These helpers are intentionally small, dependency-light, and CPU-safe so they
can be unit-tested without a GPU. See ``CLUSTER_RUN.md`` for usage.
"""
