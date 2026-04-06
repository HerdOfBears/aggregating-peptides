from setuptools import setup, find_packages

setup(
    name="aggrepep",
    version="0.1.0",
    author="Jyler Menard",
    author_email="jyler.menard@mail.concordia.ca",
    description="A package for aggrepep. This package provides tools for setting up, simulating (aa and cg), and analyzing peptide aggregation simulations.",
    packages=find_packages(),
    install_requires=[
        "openmm",
        "martini_openmm",
        "numpy",
        "pandas",
        "biopython",
        "scipy",
        "matplotlib",
        "seaborn",
        "vermouth",
        "PeptideBuilder",
        "peptides",
        "pdbfixer", # note: you may need to clone the pdbfixer repo and install it manually
        "freesasa",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)