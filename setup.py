from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this functions will return the list of requiremts
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        #as readline rea \n for new line we should remove that
        requirements= [req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

        return requirements


setup(
    name='MLproject',
    version='0.0.1',
    author='Atharva',
    author_email='master.atharva04@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)