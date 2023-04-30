import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qatrah",
    version="0.1.0",
    author="[Anas], [Basant], [Mohammed], [Airin], [Lakshika], [Sanjana], [Selin Doga], [Yaser], [Fouad], [El Amine], [Victory], [Akash Kant]",
    author_email="akant1@asu.edu",
    description="Using quantum computing to design a more precise, environmental friendly and robust water distribution network and debugging.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qcswat/qatrah",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
    keywords="WDN, Quantum Computing, QUBO, QML, Optimization",
    install_requires=[
        "qiskit",
        "pennylane"
        # other required packages
    ],
    project_urls={
        "Documentation": "https://github.com/qcswat/qatrah",
        "Source Code": "https://github.com/qcswat/qatrah",
        "Tutorials": "https://github.com/qcswat/qatrah/tree/main/tutorials_examples",
    },
)
