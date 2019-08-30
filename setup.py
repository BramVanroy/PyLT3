from setuptools import setup, find_packages

with open('README.rst', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='pylt3',
    version='0.7.9',
    description='A collection of helper functions and NLP scripts',
    long_description=long_description,
    keywords='nlp xml file-handling helpers',
    packages=find_packages(exclude=('tests/', 'tests/**/*', 'tests', 'examples/', '.git')),
    url='https://github.com/BramVanroy/pylt3',
    author='Bram Vanroy',
    author_email='bramvanroy@hotmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/BramVanroy/pylt3/issues',
        'Source': 'https://github.com/BramVanroy/pylt3'
    },
    python_requires='>=3.6'
)
