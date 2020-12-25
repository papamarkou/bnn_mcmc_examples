from codecs import open
from os import path
from setuptools import find_packages, setup

from bnn_mcmc_examples import __version__

url = 'https://github.com/papamarkou/bnn_mcmc_examples'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bnn_mcmc_examples',
    version=__version__,
    description='Examples of MCMC sampling for Bayesian neural networks',
    long_description=long_description,
    url=url,
    download_url='{0}/archive/v{1}.tar.gz'.format(url, __version__),
    packages=find_packages(),
    license='MIT',
    author='Theodore Papamarkou',
    author_email='theodore.papamarkou@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'
    ],
    keywords=['Bayesian', 'MCMC', 'Monte Carlo', 'neural networks'],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.2', 'pandas', 'scikit-learn', 'torch>=1.6.0', 'eeyore>=0.0.16', 'kanga>=0.0.17', 'matplotlib>=3.3.3'
    ],
    package_data={'bnn_mcmc_examples': ['data/*/x.csv', 'data/*/y.csv', 'data/*/readme.md']},
    include_package_data=True,
    zip_safe=False
)
