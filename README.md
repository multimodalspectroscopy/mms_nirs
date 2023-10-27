# MMS NIRS

## What

This repo contains Python code that can be used with the NIRS data generated at the multimodal spectroscopy group at UCL.

## Installation

To install we use [`conda`](https://docs.conda.io/en/latest/) and [`poetry`](https://www.google.com/search?q=poetry+python&oq=poetry+python&sourceid=chrome&ie=UTF-8). Installation instructions for each of these can be found in their respective documentation. We use these tools to make it easy to develop our Python code in isolated environments (using `conda`) and to easily add, modify and publish packages (using `poetry`).

To create a new environment start by making a new conda environment with

```bash
conda env create -f environment.yml
```
which will create an environment based on the `environment.yaml` in this repository. The name of the environment will match the one in this file.]

Once this is created you can activate the environment in order to use it

```bash
conda activate mms_nirs
```

Once the environment is activated you can then install all the dependencies of the project with `poetry install`.

To add a new depdency, you use 
```bash
poetry add <package_name> # for dependencies that should be available in the final package e.g. numpy
poetry add --group dev <package_name> # for dependencies that are only for development e.g. pytest, black etc.
```
and so on as per the documentation. This should install dependencies based on the `conda-forge` channel which is generally more reliable for data science and scientific computing code, and then it will fal back to PyPi if it's not available there.


## Publishing

To publish a new package version you'll need to configure `poetry` locally as per [the documentation](https://python-poetry.org/docs/repositories/#configuring-credentials).
Once that's done you can use the `Makefile` to publish the changes.

First create a new version of the code in Github

```bash
make version v=X.X.X
```
where `X.X.X` is the version as defined by semver (`major.minor.patch`).

With that done you can then push the change.

To push to the production PyPi repo use 

```bash
make publish-prod
```
and to publish to the test repo (if you want to test changes without impacting other users of the package) use

```bash
make publish-test
```

Once that's done you can change the version anywhere it's needed to use the new code.
