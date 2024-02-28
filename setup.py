# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ivon-opt",
    version="0.1",
    url="https://github.com/team-approx-bayes/ivon",
    packages=find_packages(),
    description="An optimizer for neural networks based on variational learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    license='GPLv3+',
    author='IVON Team',
    author_email='ivonteam@googlegroups.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
    ],
)

