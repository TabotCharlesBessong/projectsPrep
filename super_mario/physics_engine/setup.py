from setuptools import setup, find_packages

setup(
    name="game_physics",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
    ],
    author="Charles",
    author_email="example@example.com",
    description="A physics engine library for game development",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/game_physics",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.8',
)