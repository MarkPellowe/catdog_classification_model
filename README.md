# Cat/dog Classification Model

Repository containing a pre-trained computer vision model for classifying images of cats and dogs, to be included in the Scivision catalog.

## Installation

To install this package into your environment, run pip install . from this directory.

## Contents of this package

The python code itself is in the directory `catdog_classification_model/`

- `__init__.py` just allows this directory to be treated as a python package, and imports the contents of `model.py` into the package's namespace.
- `model.py` this is the wrapper for the model itself. It ensures that images on which the model will be run are the correct size, and it provides the interface that Scivision expects. It should contain a single class, which has a `predict()` method. It should also have a way of loading the weights.
