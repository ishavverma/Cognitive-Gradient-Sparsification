from setuptools import setup, find_packages

setup(
    name='cgs',
    version='0.1.0',
    description='Cognitive Gradient Sparsification: A Self-Sufficient Data-Efficient Learning Framework',
    author='CGS Research',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
    ],
    extras_require={
        'viz': ['matplotlib>=3.5.0'],
        'export-pytorch': ['torch>=1.9.0'],
        'export-tensorflow': ['tensorflow>=2.8.0'],
        'all': ['matplotlib>=3.5.0', 'torch>=1.9.0', 'tensorflow>=2.8.0'],
    },
)
