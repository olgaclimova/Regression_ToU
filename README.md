## Project: Predicting Daily NO₂ in Milan (2023–2024)

Problem statement

We aim to predict next-day mean NO₂ concentration in Milan using recent pollution and weather data. Under current EU law the annual limit for NO₂ is 40 µg/m³, with reforms underway to tighten this to 20 µg/m³ by 2030; the WHO 2021 guideline is even stricter at 10 µg/m³. Forecasting can therefore support compliance and public-health protection.

Hypothesis

H1. Higher traffic intensity (Area C gate entries) is positively associated with higher daily mean NO₂ the same day.

H2. Wind speed and precipitation are negatively associated with daily NO₂ (dispersion/deposition effects).

Target & features

Target (y): Daily mean NO₂ (µg/m³) for city of Milan. Source: www.dati.comune.milano.it

Features:

Daily Wind speed & direction
Daily Precipitation
Daily Air temperature
Daily Traffic: Area C entries
Sources: www.ilmeteo.it www.dati.comune.milano.it
