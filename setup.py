from setuptools import setup

# Core dependencies (always required)
CORE_DEPS = [
    'tqdm>=4.66.1',
    'numpy>=1.26.4',
    'simple_parsing>=0.1.5',
    'mosaicml-streaming>=0.7.5',
    'sentencepiece>=0.1.99',
    'zstandard>=0.23.0',
    'universal-pathlib>=0.2.2',
    'fsspec[s3]>=2023.1.0',
    # 'transformers==4.39.3',
]

# Optional dependencies for dataset support
DATASET_DEPS = [
    'datasets>=2.18.0',
]

setup(
    name='datatools-py',
    version='0.2',
    packages=['datatools'],
    install_requires=CORE_DEPS,
    extras_require={
        'datasets': DATASET_DEPS,
        'full': DATASET_DEPS,  # Alias for datasets
    },
    author='Alexander Wettig',
    description='Library and scripts for common LM data utilities (tokenizing, splitting, packing, ...)',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/CodeCreator/datatools',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'peek=datatools.scripts.peek:main',
            'merge_index=datatools.scripts.merge_index:main',
            'pack=datatools.scripts.pack:main',
            'wrangle=datatools.scripts.wrangle:main',
            'tokenize=datatools.scripts.tokenize:main',
        ]
    },
    python_requires='>=3.6',
)
