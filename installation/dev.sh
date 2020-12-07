#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export PKGNAME='bnn_mcmc_examples'
export PYVERSION='3.7'
export CONDADIR="$HOME/opt/continuum/miniconda/miniconda3"
export PYPKGDIR="$HOME/opt/python/packagesss"
export CONDAENV="$CONDADIR/envs/$PKGNAME"
export CONDABIN="$CONDADIR/bin/conda"
export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'

declare -a PKGNAMES=('kanga' 'eeyore' 'bnn_mcmc_examples')

pkgurl () {
  echo "https://github.com/papamarkou/$1.git"
}

pkgdevreqs () {
  echo "$PYPKGDIR/$1/installation/requirements_$1.txt"
}

sudo apt-get update

sudo apt-get install tree

wget https://repo.anaconda.com/miniconda/$CONDASCRIPT
chmod u+x $CONDASCRIPT

$SHELL $CONDASCRIPT -b -p $CONDADIR

$CONDABIN create -n $PKGNAME -y python=$PYVERSION

$CONDABIN init $(basename $SHELL)
$CONDABIN config --set auto_activate_base false

mkdir -p $PYPKGDIR

for pkg in "${PKGNAMES[@]}"
do
   git -C $PYPKGDIR clone $(pkgurl $pkg)
   $CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/$pkg -r $(pkgdevreqs $pkg)
done

rm $HOME/$CONDASCRIPT
