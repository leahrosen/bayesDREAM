"""
Setup script for bayesDREAM package.

Installation:
    pip install -e .        # Development mode (editable)
    pip install .           # Standard installation
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'bayesDREAM', '__init__.py')
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.1.0'

# Read long description from README if it exists
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='bayesdream',
    version=get_version(),
    author='bayesDREAM Development Team',
    description='Bayesian framework for modeling CRISPR perturbation effects across multiple molecular modalities',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',  # Required for SplineTransformer and Ridge
        'torch>=2.2.0',
        'pyro-ppl>=1.9.0',
        'matplotlib>=3.7',
        'seaborn>=0.12',
        'adjustText>=1.0.0',  # Non-overlapping label placement for volcano plots
        'h5py>=3.8.0',
        'typer>=0.12.0',
        'pyyaml>=6.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'notebook>=6.0.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points={
        'console_scripts': [
            'bayesdream=bayesDREAM.cli:main',
        ],
    },
)
