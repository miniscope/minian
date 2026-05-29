import setuptools

NAME = "minian"
REQUIREMENTS = "requirements/requirements-base.txt"

with open("README.md", "r") as fh:
    long_description = fh.read()


def _get_requirements(requirement_file):
    with open(requirement_file) as f:
        reqs = []
        for l in f.read().splitlines():
            reqs.append(l)
        return reqs


setuptools.setup(
    name=NAME,
    version="1.2.1",
    author="MiniAn Developers",
    author_email="denisecailab@gmail.com",
    description="MiniAn is an analysis pipeline and visualization tool inspired by both CaImAn and MIN1PIPE package specifically for Miniscope data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeniseCaiLab/minian",
    install_requires=_get_requirements(REQUIREMENTS),
    packages=setuptools.find_packages(),
    project_urls={
        "Documentation": "https://minian.readthedocs.io/",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "minian-install = minian.install:main",
        ],
    },
    python_requires=">=3.9",
)
