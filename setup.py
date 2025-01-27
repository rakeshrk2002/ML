from setuptools import find_packages,setup
from typing import List

e_hyphen = 'e .'

def get_requirements(file_path):
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('-e')]
    return requirements

setup(
    name='ML_Project',
    version='0.0.1',
    author='Rakesh',
    author_email='rakeshthiagu2002@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)