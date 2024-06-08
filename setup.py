from setuptools import setup

setup(
    name='datatools',
    version='0.1',
    packages=['datatools'],
    install_requires=[
        'tqdm==4.66.1',
        'numpy==1.26.4',
        'simple_parsing==0.1.5',
        'mosaicml-streaming==0.7.5',
        'datasets==2.18.0',
        'sentencepiece==0.1.99',
        # 'transformers==4.39.3',
    ],
    entry_points={
        'console_scripts': [
            'dump=datatools.scripts.dump:main',
            'merge=datatools.scripts.merge:main',
            'pack=datatools.scripts.pack:main',
            'splits=datatools.scripts.splits:main',
            'tokenize=datatools.scripts.tokenize:main',
        ]
    },
    python_requires='>=3.6',
)