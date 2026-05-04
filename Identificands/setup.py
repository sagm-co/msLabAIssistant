from setuptools import setup
from pathlib import Path
import platform,sys

if platform.system() != 'Linux':
    sys.stderr.write("Currently Identificands is only supported on Linux.\n")
    sys.exit(1)

wd = Path(__file__).parent
long_description = (wd / "README.md").read_text()

setup(
    name="Identificands",
    version="0.0.1",
    install_requires=["ipython >= 8.18.1",
	              "pillow >= 9.0.1",
	              "joblib >= 1.3.2",
	              "mordred == 1.2.0",
	              "rdkit==2023.9.2",
	              "scipy >= 1.15.2",
	              "scikit-image >= 0.22.0",
	              "scikit-learn >= 1.3.2",
	              "tqdm >= 4.66.1",
	              "python-intervals >= 1.10.0",
	              "matplotlib  >= 3.10.1",
	              "numpy  >= 1.26.4",
	              "pandas >= 2.2.3",
	              "tensorflow >= 2.15.0",
                      "jupyter >= 1.1.1",
                      "jupyterlab >= 4.2.6",
            ],
    platforms=['Linux'], 
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    author="Sergio A. Gonzalez-Monico",
    author_email="sagonzalezm@unal.edu.co",
    license="GPL-3.0-or-later",
    description="AI-powered tool to assist compound identification",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/sagm-co/msLabAIssistant",
    python_requires=">=3.10",
) 
