#!/bin/bash

# Start up script for setting up environment on Ubuntu 20.04 LTS

export METAUSER='theodore'
export BASEDIR="/home/$METAUSER"
export BINDIR="$BASEDIR/bin"
export PKGNAME='bnn_mcmc_examples'
export PYVERSION='3.7'
export CONDADIR="$BASEDIR/opt/continuum/miniconda/miniconda3"
export PYPKGDIR="$BASEDIR/opt/python/packages"
export CONDAENV="$CONDADIR/envs/$PKGNAME"
export CONDABIN="$CONDADIR/bin/conda"
export CONDASCRIPT='Miniconda3-latest-Linux-x86_64.sh'
export KANGAURL='https://github.com/papamarkou/kanga.git'
export EEYOREURL='https://github.com/papamarkou/eeyore.git'
export PKGURL="https://github.com/papamarkou/$PKGNAME.git"
export KANGADEVREQS="$PYPKGDIR/kanga/requirements.txt"
export EEYOREDEVREQS="$PYPKGDIR/eeyore/installation/requirements.txt"
export PKGDEVREQS="$PYPKGDIR/$PKGNAME/installation/requirements.txt"

sudo apt-get update

sudo apt-get install tree

su - $METAUSER -c "wget https://repo.anaconda.com/miniconda/$CONDASCRIPT"
su - $METAUSER -c "chmod u+x $CONDASCRIPT"

su - $METAUSER -c "$SHELL $CONDASCRIPT -b -p $CONDADIR"

su - $METAUSER -c "$CONDABIN create -n $PKGNAME -y python=$PYVERSION"

su - $METAUSER -c "$CONDABIN init $(basename $SHELL)"
su - $METAUSER -c "$CONDABIN config --set auto_activate_base false"

su - $METAUSER -c "mkdir -p $BINDIR"
su - $METAUSER -c "ln -s $CONDABIN $BINDIR"
su - $METAUSER -c "echo \"export PATH=$BINDIR:$PATH\" >> $BASEDIR/.bashrc"

su - $METAUSER -c "mkdir -p $PYPKGDIR"

su - $METAUSER -c "git -C $PYPKGDIR clone $KANGAURL"
su - $METAUSER -c "$CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/kanga -r $KANGADEVREQS"

su - $METAUSER -c "git -C $PYPKGDIR clone $EEYOREURL"
su - $METAUSER -c "$CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/eeyore -r $EEYOREDEVREQS"

# su - $METAUSER -c "git -C $PYPKGDIR clone $PKGURL"
# su - $METAUSER -c "$CONDABIN run -p $CONDAENV pip install -e $PYPKGDIR/$PKGNAME -r $PKGDEVREQS"

su - $METAUSER -c "rm $BASEDIR/$CONDASCRIPT"
