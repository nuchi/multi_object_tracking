from setuptools import setup, find_packages

NAME = 'multi_object_tracking'

install_requires = [
    'filterpy>=1.4.5',
    'importlib-resources; python_version < "3.7"'
    'matplotlib>=3.2.0',
    'numpy>=1.16',
    'opencv-python>=4.0.0',
    'scipy>=1.5',
    'setuptools',
    'tqdm>=4',
]

setup(
    name=NAME,
    version='0.0.1',
    packages=find_packages(include=(NAME, f'{NAME}.*')),
    package_data={NAME: ['default_params.json']},
    install_requires=install_requires,
    description='Multi-object tracking of dark blotches on light background',
    author='Haggai Nuchi',
    author_email='haggai@haggainuchi.com',
)
