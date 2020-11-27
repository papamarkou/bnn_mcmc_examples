# Steps for manual installation

```
PYBIN="python3"
PKGDIR="${HOME}/opt/python/packages"

conda update --all

conda create -n bnn_mcmc_examples python=3.6

conda activate bnn_mcmc_examples

conda install pytorch torchvision cpuonly -c pytorch # Linux
# conda install pytorch torchvision -c pytorch # Mac

# Optional additional packages: scikit-learn, matplotlib, seaborn, spyder

cd ${PKGDIR}
git clone git@github.com:papamarkou/eeyore.git
cd eeyore
${PYBIN} setup.py develop --user

cd ${PKGDIR}
git clone git@github.com:papamarkou/kanga.git
cd kanga
${PYBIN} setup.py develop --user

cd ${PKGDIR}
git clone git@github.com:papamarkou/bnn_mcmc_examples.git
cd bnn_mcmc_examples
${PYBIN} setup.py develop --user
```
