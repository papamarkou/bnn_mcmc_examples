#!/bin/bash

export PKGNAME='bnn_mcmc_examples'
export CONDADIR="$HOME/opt/continuum/miniconda/miniconda3"
export CONDAENV="$CONDADIR/envs/$PKGNAME"
export CONDABIN="$CONDADIR/bin/conda"
export OUTPUTPATH="$HOME/output/bnn_mcmc_examples/mlp/penguins/hmc"

qsub \
  -cwd \
  -V \
  -l short \
  -N mlp_penguins_hmc \
  -o $OUTPUTPATH \
  -e $OUTPUTPATH \
  $CONDABIN run -p $CONDAENV python benchmark_run.py
