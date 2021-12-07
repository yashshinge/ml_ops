# simple-classifier

----
Simple ML classifier.

## Install requirements
```
pip install -r requirements.txt
```

## Train the model
```
python train.py

optional arguments:
--epochs                 Number of epochs to train
--seed                   Random seed
```

## Run unit tests
```
pytest -v
```

## Suggestions to increase reliability
- Develop modular code for repetitive jobs like data pre-processing, and post-processing.
- Parametrize constants to avoid frequent changes to main framework:
    - Define the model training configuration in a separate ```config.py``` or a ```config.yaml``` file, and load it in the script.
    - Same for defining input data path, model output path, and so.
    - Passing arguments to the script directly (added epochs and seed as optional args to ```train.py``` as an example in this repo).
- Create in-house packages for common tasks like connecting to a database, common utils, etc., to save time and help research scientists focus.
- Build tools to automate folder structure (framework) creation.
- Enable model versioning and tracking.
