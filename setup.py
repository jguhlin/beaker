import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bio-beaker",
    version="0.0.1",
    author="Joseph Guhlin",
    author_email="joseph.guhlin@gmail.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jguhlin/beaker",
    packages=setuptools.find_packages(),
    keywords='genomics, machine learning, tensorflow',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Python Software Foundation License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        'Development Status :: 4 - Beta',
    ],
    install_requires=[
        "numpy",
        "numba",
        "cyvcf2",
        "sparse",
    ],
    python_requires='>=3.6',
    project_urls={
        'Bug Reports': 'https://github.com/jguhlin/beaker/issues',
        'Source': 'https://github.com/jguhlin/beaker',
        'Paper': 'https://www.biorxiv.org/'
    },
)