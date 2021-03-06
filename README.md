# TDT4173-project
The goal of this project is to use the [yelp-dataset](https://www.kaggle.com/yelp-dataset/yelp-dataset) to predict the number of stars given for a review based on the words used in the review text. We will use word embeddings try different types of machine learning techniques, namely kNN with bag-of-words and LSTM with word embeddings.

The project is hosted on github at <https://github.com/SimenHolmestad/TDT4173-project>.

## About the dataset
The Yelp dataset (<https://www.kaggle.com/yelp-dataset/yelp-dataset>) contains a wealth of information from Yelp, a popular website for crowd-sourced reviews of various physical establishments. The dataset contains several categories, of which we are interested in one: The reviews. This data set alone is 5.9GB in size, and contains 5,200,000 user reviews. This should be more than enough to train and test our models.

**Note:** The kaggle website claims that the dataset contains 5,200,000 user reviews. However, when running
```
grep review_id yelp_academic_dataset_review.json | sort | uniq | wc -l
```

the result says that there are 8,021,122 unique lines in the file all containing `review_id` , so in reality, it seems like there is a bit more than 8 million reviews in the dataset.

# Requirements
To install the required python packages needed for the project, run
```
pip3 install -r requirements.txt
```
from the root folder of the project.

# Repo structure
Information about folders and important files in the project is found below.

## The Preprocessing Folder
The preprocessing folder contains python scripts for data preprocessing. These includes:

- `preprocess_review_data.py`: A script for processing the raw data from the yelp dataset. When running the script, `file-path-to-original-file` and `output-filename` are required arguments.
- `reduce_dataset.py`: A script for reducing the dataset to a more manageable format for testing. When running the script, `file-path-to-original-file` and `output-filename` are required arguments.
- `create_plots.py`: A script for creating plots from a file containing preprocessed data. When running the script, `file-path-to-data-file` is a required argument.

## The Plots Folder
This folder contains plots from running `create_plots.py`.

## The Data folder
The entire dataset is not hosted in the repo because it is too large for GitHub file size contstraints. Instead, a file containing the 100,000 first lines in the dataset can be found at `Data/first_100000_reviews.json`, and a file containing the first 100,000 processed lines after running `Preprocessing/preprocess_review_data` can be found at `Data/first_100000_processed_reviews.json`.

## The Source folder
The source folder contains python scripts for model training. These include:
- `lstm_model_training.py`: A script for training the lstm model. When running the script, `file-path-to-data-file` and `directory-for-output-files` are required arguments.

In addition, the Source folder contains the Google Cloud Function used as a backend for the website created for the project.

## The Results folder
The Results folder contains plots showing how the models have performed.

## The Model folder
The Model folder contains models created from running the scripts in the source folder.

# Website
The group has created a webpage to demonstrate how the models perform on novel data. The web page is currently hosted with GitHub Pages and can be found at <https://simenholmestad.github.io/TDT4173-webpage>.

The backend for the website is a Google Cloud Function running an LSTM model. As the prediction is done directly in the cloud function, the backend is a little bit slow. The LSTM model for the backend is trained on an evenly balanced dataset using 2 epochs.

## Deploying the cloud function to google cloud
The entire backend of the website can be found in `source/cloud_function`. To export the function to google cloud using the cloud sdk, run the command:
```
gcloud functions deploy function-1 --trigger-http --runtime python38 --allow-unauthenticated --region europe-west1 --memory 4096MB --entry-point main
```
from the `source/cloud_function`-directory.
**NOTE** You must be logged into Google Cloud in your terminal and have access to the GCP project.

# Report
The OverLeaf link to the report can be found here: <https://www.overleaf.com/9135544627bjtbdxctxqhm>.

# Useful links
- Resources from the course staff: <https://github.com/ntnu-ai-lab/tdt4173-2020>
