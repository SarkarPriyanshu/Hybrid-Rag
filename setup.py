from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
        # Ignore pip options such as --extra-index-url
        return [line for line in lines if line.strip() and not line.startswith("--")]

requirements = parse_requirements("requirements.txt")

setup(
    name="rag-app",
    version="0.1",
    author="priyanshu",
    packages=find_packages(),
    install_requires=requirements,
)
