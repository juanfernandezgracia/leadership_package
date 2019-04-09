import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="leadership_package_juanfernandezgracia",
    version="0.0.1",
    author="Juan Fernandez-Gracia",
    author_email="juanfernandez1984@gmail.com",
    description="A package for detecting follower-followee relations in point (acoustic) data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
