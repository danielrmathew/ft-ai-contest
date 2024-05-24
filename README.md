# ft-ai-contest

## Overview

This is our 2024 Franklin Templeton University AI Competition entry.

## File Structure

The `components` folder contains two other folders: `modules` and `notebooks`. The `modules` folder contains .py files from which `full_pipeline.ipynb` imports the necessary models. `notebooks` contains the Jupyter notebooks used to develop the models.

The `articles` folder houses the Yahoo Finance articles that have been collected for training. 

The `reference` folder contains other work and testing in previous files.

## System Configuration

Run the following command in your terminal to install the necessary packages for this project:

```pip install -r requirements.txt```

## Run model

Open full_pipeline.ipynb and run the cells to load the LLM into memory and begin inference.

NOTE: Only load the model using `get_llm` once! Running it multiple times will store multiple models in memory and cause an OutofMemory error.

The `user_prefs` variable can be modified to encapsulate your preferences in portfolio generation. The valid values for "risk" are ["conservative", "moderate", "aggressive"]. For the "industries" parameter, valid options are ["Energy", "Materials", "Industrials", "Utilities", "Healthcare", "Financials", "Consumer Discretionary", "Consumer Staples", "Technology", "Communication Services", "Real Estate"].


