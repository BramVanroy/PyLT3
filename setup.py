from setuptools import setup

setup(
    name='pylt3',
    version='0.6.0',
    description='A collection of helper functions and NLP scripts',
    long_description='During my time working on the PhD project PreDict, I have written and gathered a bunch of useful'
                     ' functions. They are collected here as part of the pylt3 package.',
    keywords='nlp xml file-handling helpers',
    packages=['pylt3'],
    url='https://github.com/BramVanroy/pylt3',
    author='Bram Vanroy',
    author_email='bramvanroy@hotmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Text Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    project_urls = {
        'Bug Reports': 'https://github.com/BramVanroy/pylt3/issues',
        'Source': 'https://github.com/BramVanroy/pylt3',
    },
    python_requires='>=3.6'
)
