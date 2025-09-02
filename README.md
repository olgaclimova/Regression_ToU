## Project: Predicting Daily NO2 in Milan

Problem statement

We aim to predict next-day mean NO₂ concentration in Milan using recent pollution and weather data. Under current EU law the annual limit for NO2 is 40 µg/m³, with reforms underway to tighten this to 20 µg/m³ by 2030. Forecasting can therefore support compliance and public-health protection.

Hypothesis

H1. Higher traffic intensity (Area C gate entries) and higher air temperature are positively associated with higher daily mean NO2 the same day.

H2. Wind speed and precipitation are negatively associated with daily NO2 (dispersion/deposition effects).

## Repository Structure

data/

raw/ → original datasets

preprocessed/ → intermediate datasets after cleaning steps

processed/ → final datasets used for model training

notebooks/ → main jupyter notebook with results, plots, and explanations

utilities/ → helper scripts (to be run inside the notebook)

results/ → json with tuned hyperparameters

## Workflow:

Open and explore notebooks/main_project.ipynb
