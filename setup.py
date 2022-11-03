from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="aylien_datascience_demos",
    version=version,
    packages=["aylien_datascience_demos"],
)
