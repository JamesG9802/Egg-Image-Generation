# Egg Image Generation
 A supervised egg image generator in Python.



# Installation and Setup

This assumes a basic understanding of terminal commands, git, and Python.

1. Clone the repository.
```
git clone git@github.com:JamesG9802/Egg-Image-Generation.git
cd Egg-Image-Generation
```

2. This project uses Poetry for dependency and project management. 
[See Poetry's page for getting started.](https://python-poetry.org/docs/#installing-with-pipx). 
Alteranatively, you can install directly from the `requirements.txt` file. 
[See Python's page for getting started.](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#using-a-requirements-file)

Installing with Poetry:
```
poetry install
```

Installing with requirements:
```
python -m pip install -r requirements.txt
```

To run Python files:

```
poetry run python src/egg_image_generation/train.py
```

With a virtual or global environment:

```
python src/egg_image_generation/train.py
```

# Running the Project

First, you need to fetch the dataset. You can either download it directly through [https://www.kaggle.com/datasets/abdullahkhanuet22/eggs-images-classification-damaged-or-not] or by running `fetch_dataset.py`

Example to save dataset to the 'dataset/' folder:
```
python src/egg_image_generation/fetch_dataset.py dataset
```

Afterwards, you can train the model with `train.py`:
```
python src/egg_image_generation/train.py 'dataset/Eggs Classification'
```
