#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export PKGNAME='bnn_mcmc_examples'
export PYVERSION='3.7'
export CONDADIR="$HOME/opt/continuum/miniconda/miniconda3"
export PYPKGDIR="$HOME/opt/python/packages"
export CONDAENV="$CONDADIR/envs/$PKGNAME"
export CONDABIN="$CONDADIR/bin/conda"
export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'
export KANGAURL='https://github.com/papamarkou/kanga.git'
export EEYOREURL='https://github.com/papamarkou/eeyore.git'
export PKGURL="https://github.com/papamarkou/$PKGNAME.git"
# export PKGURL="git@github.com:papamarkou/$PKGNAME.git"
export KANGADEVREQS="$PYPKGDIR/kanga/requirements.txt"
export EEYOREDEVREQS="$PYPKGDIR/eeyore/installation/requirements.txt"
export PKGDEVREQS="$PYPKGDIR/$PKGNAME/installation/requirements.txt"

# sudo apt-get update

# sudo apt-get install tree

wget https://repo.anaconda.com/miniconda/$CONDASCRIPT
chmod u+x $CONDASCRIPT

$SHELL $CONDASCRIPT -b -p $CONDADIR

$CONDABIN create -n $PKGNAME -y python=$PYVERSION

$CONDABIN init $(basename $SHELL)
$CONDABIN config --set auto_activate_base false

mkdir -p $PYPKGDIR

git -C $PYPKGDIR clone $KANGAURL
$CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/kanga -r $KANGADEVREQS

git -C $PYPKGDIR clone $EEYOREURL
$CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/eeyore -r $EEYOREDEVREQS

git -C $PYPKGDIR clone $PKGURL
$CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/$PKGNAME -r $PKGDEVREQS

rm $HOME/$CONDASCRIPT
