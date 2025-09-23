# Sales Monitoring

An interactive Streamlit web app for monitoring and analyzing COMPANY sales, invoices, and client trends across multiple years.

## Overview

This app consolidates payment monitoring workbooks (2019–present) into a single interactive dashboard.
It was designed to help company staff and management:

Track total sales and monthly trends across years.

Identify top-performing clients and their shares of total sales.

Visualize invoice type mix (Order Slip vs. Charge Inv) over time, both as absolute values and normalized percentages.

Monitor client base growth with active companies per year.

Perform STL decomposition on total sales to reveal long-term trend vs. seasonality.

Export CSV summaries for further analysis.

All charts are interactive (zoom, hover tooltips, download as PNG) and update instantly based on filters.

## Key Features

Multi-year consolidation: Combines data from multiple Excel macro workbooks automatically.

Deduplication: Removes duplicates across files based on receipt number and date.

Canonical company names: Reduces aliasing so the same client is counted once across years.

Dynamic filters: Choose years, receipt types, and number of top companies.

Top-N charts and pies: Quickly see your top clients or invoice types.

Client growth tracking: “Active companies per year” barplot and KPI.

Seasonality analysis: STL trend component for forecasting and planning.

## Requirements

Python 3.11 (or newer)

Packages: streamlit, pandas, numpy, openpyxl, plotly, statsmodels

Use the included requirements.txt to install dependencies.

## Data Privacy

This app is designed to run locally. Your data never leaves your machine unless you deliberately deploy it on a server.
