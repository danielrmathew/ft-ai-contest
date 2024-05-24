# ft-ai-contest

## Overview

This is our 2024 Franklin Templeton University AI Competition entry.

## System Configuration

Run the following command in your terminal to install the necessary packages for this project:
`pip install -r requirements.txt`

## Run model

Open full_pipeline.ipynb and run the cells to load the LLM into memory and begin inference.

NOTE: Only load the model using `get_llm` once! Running it multiple times will store multiple models in memory and cause an OutofMemory error.

The `user_prefs` variable can be modified to encapsulate your preferences in portfolio generation.
