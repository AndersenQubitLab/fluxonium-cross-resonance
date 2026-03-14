from setuptools import setup, Extension
from Cython.Build import cythonize


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

ext_modules = cythonize([
    Extension("fluxoniumcr.magnus", ["fluxoniumcr/magnus.pyx"]),
])

setup(
    name="fluxoniumcr",
    version="0.1",
    description="Simulation and plotting tools for the publication: Exploration of Fluxonium Parameters for Capacitive Cross-Resonance Gates.",
    author="Eugene Y. Huang",
    packages=["fluxoniumcr"],
    install_requires=requirements,
    ext_modules=ext_modules,
    zip_safe=False,
)
