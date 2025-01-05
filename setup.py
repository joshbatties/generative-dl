from setuptools import setup, find_packages

setup(
    name="generative-dl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.8.0',
        'numpy>=1.19.2',
        'matplotlib>=3.3.0',
        'scipy>=1.7.0',
        'Pillow>=8.0.0',
        'pyyaml>=5.4.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=20.8b1',
            'isort>=5.6.0',
            'flake8>=3.8.0',
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A collection of generative deep learning models implemented in TensorFlow",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/generative-dl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
