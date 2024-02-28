from setuptools import find_packages, setup

setup(
    name="marl-aquarium",
    version="0.1.10",
    packages=find_packages(exclude=["examples"]),
    license="MIT",
    description="Aquarium: A Comprehensive Framework for Exploring Predator-Prey Dynamics through Multi-Agent Reinforcement Learning Algorithms",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Yannick Erpelding and Michael Kölle",
    author_email="michael.koelle@ifi.lmu.de",
    url="https://github.com/michaelkoelle/marl-aquarium",
    keywords=[
        "artificial intelligence",
        "pettingzoo",
        "multi-agent",
        "reinforcement learning",
        "deep learning",
        "predator-prey",
        "gymnasium",
        "gym",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium==0.28.1",
        "moviepy==1.0.3",
        "numpy==1.22.4",
        "pettingzoo==1.24.2",
        "pygame==2.1.3",
    ],
    include_package_data=True,
)
