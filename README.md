# Project Documentation: Solar Radiation Data Analysis

- The project focuses on the exploratory data analysis (EDA) of solar radiation measurement data collected from Benin, Sierra Leone, and Togo. The primary goal is to derive insights from the dataset to inform strategic decisions for solar energy investments. The project involves setting up a development environment, performing EDA, conducting statistical analysis, and reporting the findings.

# Folder Structure

- The project is organized as follows:
  ├── .vscode/
  │ └── settings.json # VSCode settings for consistent development environment
  ├── .github/
  │ └── workflows/
  │ ├── unittests.yml # CI/CD workflow for running unittests
  ├── .gitignore # Git ignore file to exclude unnecessary files from the repository
  ├── requirements.txt # Python dependencies for the project
  ├── README.md # Overview and instructions for the project
  ├── src/
  │ ├── **init**.py # Init file for the src module
  │ ├── eda.py # EDA functions for data analysis
  ├── notebooks/
  │ ├── **init**.py # Init file for the notebooks module
  │ ├── eda.ipynb # Jupyter notebook for interactive EDA
  ├── tests/
  │ ├── **init**.py # Init file for the tests module
  │ ├── test_eda.py # Unit tests for the EDA functions
  └── scripts/
  ├── **init**.py # Init file for the scripts module
  └── run_eda.sh # bash Script to execute EDA and generate reports

# Development Environment Setup

- Python Environment:

* A virtual environment was set up to manage dependencies. The required packages are listed in requirements.txt and include libraries like pandas, numpy, matplotlib, and seaborn.

- Version Control:

* Git was used for version control, and the repository was hosted on GitHub. The repository followed a branch-based workflow with branches for different tasks (e.g., task-1 for initial EDA).

- CI/CD:

* A GitHub Actions workflow (unittests.yml) was set up to automatically run tests whenever code is pushed to the repository, ensuring code quality.

# Exploratory Data Analysis (EDA)

## The EDA was conducted using the eda.py script, which includes several key functions:

- Summary Statistics: Calculated mean, median, standard deviation, and other statistical measures for each numeric column.
- Data Quality Check: Checked for missing values, outliers, and incorrect entries in critical columns.
- Time Series Analysis: Analyzed how variables like GHI, DNI, DHI, and Tamb change over time.
- Correlation Analysis: Determined correlations between different variables, such as solar radiation components and temperature measures.
- Wind Analysis: Explored wind speed and direction data to identify trends.
- Temperature Analysis: Compared module temperatures with ambient temperature.
- Visual Analysis: Generated histograms, box plots, and scatter plots to visualize the data distribution and relationships.
- Data Cleaning: Cleaned the dataset by handling anomalies and missing values.
- Z-Score Analysis: A Z-score analysis was performed to detect outliers in the dataset. The detect_outliers() function in eda.py calculates Z-scores for each data point and flags those that significantly deviate from the mean.

# Notebook for Interactive EDA

An interactive Jupyter notebook (eda.ipynb) was created to allow for exploratory analysis in an interactive environment. The notebook provides a step-by-step EDA process with visualizations and commentary.

# Unit Testing

Unit tests were implemented in the tests/test_eda.py file to ensure that the EDA functions work as expected.

# Execution Script

The scripts/run_eda.sh bash script automates the EDA process. Running this script performs the analysis on the dataset and generates a report of the findings.

# Contributions

- Environment Setup: Configured the Python environment, set up Git version control, and implemented CI/CD workflows.
- EDA Development: Developed the EDA functions in eda.py, conducted a thorough analysis, and cleaned the data.
- Statistical Analysis: Performed Z-score analysis to detect outliers in the dataset.
- Interactive Analysis: Provided a Jupyter notebook for interactive EDA, allowing others to explore the data and analysis steps.

# summary

Generally, this project demonstrates a complete workflow from setting up a development environment to performing data analysis and reporting findings.
