# TDT4173-project

The goal of this project is to use the [yelp-dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset) to predict the number of stars given for a review based on the words used in the review text. We will use word embeddings try different types of machine learning techniques, namely bag-of-words and LSTMs.

The project is hosted on github at <https://github.com/SimenHolmestad/TDT4173-project>.

## About the dataset

The Yelp dataset (<https://www.kaggle.com/yelp-dataset/yelp-dataset>) contains a wealth of information from Yelp, a popular website for crowd-sourced reviews of various physical establishments. The dataset contains several categories, of which we are interested in one: The reviews. This data set alone is 5.9GB in size, and contains 5,200,000 user reviews. This should be more than enough to train and test our models.

## How to use Google Cloud AI platform

The project is developed using [a notebook on GCP AI Platform](https://console.cloud.google.com/ai-platform/notebooks/list/instances?project=tdt4173-ml-project). Please note that you need to be added to the project before trying to access it.

The VM instanced used for the notebook has 2 CPUs, 7.5 GB RAM and a nVIDIA Tesla K80 GPU, as well as persistent storage.

How to user:

1. Select the ml-project-notebook in the list and select "START" from the header. The notebook should now be starting.
2. Once the notebook has started, click "OPEN JUPYTERLAB" to access the [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/#). Work is done in the "TDT4171-project"-folder.
   NOTE: This folder is Git-enabled and connected to this repo. You can use git from either the terminal or the built-in tools in JupyterLab.
3. Once you've completed your work, return to the notebook list and stop the notebook by selecting it again and clicking "STOP". This is _*very*_ important, as it saves on Cloud Credits and allows us to use the project loinger.

## Report

The link for editing the report can be found here: <https://www.overleaf.com/9135544627bjtbdxctxqhm>.

## Useful links

- Resources from the course staff: <https://github.com/ntnu-ai-lab/tdt4173-2020>
