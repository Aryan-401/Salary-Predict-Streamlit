# Salary Prediction Analysis

This repository contains code for analyzing salary data and predicting salaries based on various factors. The code is written in Python and utilizes popular data science libraries such as pandas, matplotlib, scikit-learn, and PyTorch.

## Dataset

The analysis is performed on the "survey_results_public.csv" dataset, which contains survey responses related to various aspects of software developers, including their salaries, demographics, education level, employment details, and more.

## Code Usage

The code snippet provided performs the following steps:

1. Loading the dataset using pandas and printing its shape.
2. Preprocessing the data by selecting relevant columns and renaming the "ConvertedCompYearly" column to "Salary".
3. Removing rows with missing salary values.
4. Dropping rows with any missing values in other columns.
5. Filtering the data to include only employed, full-time respondents.
6. Encoding categorical variables (Country and Education Level) using label encoding.
7. Performing exploratory data analysis by visualizing salary distributions by country using box plots.
8. Further refining the data by removing outliers (salaries below $1000 and above $300,000).
9. Training different regression models (Linear Regression, Decision Tree, Random Forest) to predict salaries.
10. Evaluating the model performance using mean absolute error (MAE), mean squared error (MSE), and root mean squared error (RMSE).
11. Performing hyperparameter tuning using grid search with a decision tree regressor.
12. Saving the best-performing model and label encoders as a pickle file.
13. Building a neural network model using PyTorch and training it on the preprocessed data.
14. Saving the trained neural network model and label encoders as a PyTorch checkpoint file.
15. Demonstrating salary prediction using the trained models on a sample input.
16. Creating a web app using Streamlit to demonstrate the salary prediction model.

## Dependencies

The code requires the following dependencies to be installed:

- pandas
- matplotlib
- scikit-learn
- PyTorch

Install the dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```
git clone https://github.com/Aryan-401/salary-prediction.git
```
2. Navigate to the repository directory:
```
cd salary-prediction
```
3. Extract the zip file (for the .ipybn file):
```
unzip salary-prediction.zip
```
4. Run the Script:
```
streamlit run ./app.py
```
## License

This project is licensed under the MIT License.