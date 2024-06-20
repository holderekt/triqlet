from setuptools import setup, find_packages

setup(
    name="triqlet",
    version="0.1",
    description="A PyTorch and Qiskit abstraction layer for Triplet Loss and Hybrid Quantum Learning",
    author="Ivan Diliso",
    author_email='diliso.ivan@gmail.com',
    packages=find_packages(),
    python_requires='>=3.6'
)