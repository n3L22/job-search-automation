from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="job-search-automation",
    version="0.1.0",
    author="Antonela Rakipaj",
    author_email="nelarakipaj@gmail.com",
    description="AI-powered job search automation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/n3L22/job-search-automation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "jobsearch=webapp.app:main",
        ],
    },
)