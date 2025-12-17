# FoAI_project
This project was created to evaluate the usefulness of Convolutional Neural Networks (CNNs) for path planning.

Keras is used for the model and WandB integration was implemented.

## Installation

We highly recommend using a conda env for running this project. You can fully set it up using these commands:

```bash
conda create -n FoAI python=3.10 -y
conda activate FoAI
pip install -r requirements.txt
```

Note that this project assumes you use an NVIDIA GPU. If you intend to use something else make sure to change the *requirements.txt* file to suit your needs.

Make sure you are logged in to WandB by running ```wandb login``` in your terminal.

## Running the project

### Model training

The *train.py* file has a CONFIG section near the top which allows you to change the dataset and model parameters. You can use *dataset_training.py* to see how the generated paths will look with your parameters.

Run *train.py* once you're happy with your config. This will generate the dataset and train the model on it.

### Viewing the results

A *.keras* file will be generated when training finishes. You can use *predict.py* to see well how your model performs.

A model trained on a large dataset was provided with the repo. These are the config parameters used to generate it:

```py
CONFIG = {
    "grid_size": 32,
    "batch_size": 128,
    "epochs": 250,
    "train_samples": 250000,
    "val_samples": 25000,
    "learning_rate": 0.001,
    "obstacle_density": 0.4,
    "min_distance": 15
}
```
