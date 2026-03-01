from setuptools import find_packages, setup
from typing import List

# Constants
HYPEN_E_DOT = "-e ."
REQUIREMENTS_FILE_PATH = "requirements.txt"


def get_requirements(file_path:str)->List[str]:
    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)        
        
    return requirements

        
setup(
    name = "Student-performnce-indicator_end-to-end-ml-project",
    author= "sreeman",
    author_email= "sreemanbitsmech@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements(REQUIREMENTS_FILE_PATH)
)

