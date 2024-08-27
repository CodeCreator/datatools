from setuptools import setup

setup(
    name='datatools',
    version='0.1',
    packages=['datatools'],
    install_requires=[
        'tqdm>=4.66.1',
        'numpy>=1.26.4',
        'simple_parsing>=0.1.5',
        'mosaicml-streaming>=0.7.5',
        'datasets>=2.18.0',
        'sentencepiece>=0.1.99',
        'zstandard>=0.23.0'
        # 'transformers==4.39.3',
    ],
    entry_points={
        'console_scripts': [
            'peek=datatools.scripts.peek:main',
            'merge_index=datatools.scripts.merge_index:main',
            'merge_data=datatools.scripts.merge_data:main',
            'pack=datatools.scripts.pack:main',
            'splits=datatools.scripts.splits:main',
            'tokenize=datatools.scripts.tokenize:main',
        ]
    },
    python_requires='>=3.6',
)
