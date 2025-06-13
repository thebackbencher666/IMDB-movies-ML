# IMDB-movies-ML
# IMDb Top 1000 Movies Analysis

This project analyzes the IMDb Top 1000 Movies dataset to explore trends in movie popularity over time and identify the most famous movie for each year based on the number of votes.

## Dataset

The dataset used is `imdb_top_1000.csv`, which contains information about the top 1000 movies as ranked on IMDb.

## Analysis

The analysis includes:

- Data loading and initial inspection.
- Data cleaning, including handling missing values and type conversion for relevant columns (`Released_Year`, `IMDB_Rating`, `No_of_Votes`).
- Visualization of the relationship between the number of votes and the release year.
- Linear regression to model the relationship between the number of votes and the release year.
- Identification of the most famous movie (by number of votes) for each release year.
- Visualization of the most famous movies, including their release year and genre.

## Code Structure

The analysis is performed in a series of code cells:

- **Cell 1:** Loads the dataset and displays basic information and the first few rows.
- **Cell 2:** Explores column names and previews relevant columns.
- **Cell 3:** Cleans the data by dropping rows with missing values in key columns and converts data types.
- **Cell 4:** Generates a scatter plot of movie votes versus release year using `matplotlib` and `seaborn`.
- **Cell 5:** Implements a linear regression model to predict the number of votes based on the release year and plots the regression line.
- **Cell 6:** Prints the coefficient, intercept, and R-squared score of the linear regression model.
- **Cell 7:** Reloads the dataset, cleans the relevant columns, and identifies the most famous movie for each year based on votes.
- **Cell 8:** Visualizes the most famous movies from the last 20 years using a horizontal bar chart, including their release year.
- **Cell 9:** Loads the original dataset and prints the column names.
- **Cell 10:** Cleans the data including the 'Genre' column and identifies the most famous movie per year with genre information.
- **Cell 11:** Visualizes the most famous movies across all years with their release year and genre displayed on the bars.

## Dependencies

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

These libraries can be installed using pip if not already available in your environment:
## How to Run

1. Upload the `imdb_top_1000.csv` file to your Google Colab environment.
2. Open the provided Python notebook in Google Colab.
3. Run each code cell sequentially.

## Results

The analysis provides insights into:

- The trend of movie votes over the years.
- A linear model to predict the number of votes based on the release year.
- A list and visualization of the most famous movies for each year, highlighting their popularity and genre.
# Book Rating Prediction

This project aims to predict the average rating of books using various features from a dataset. The project utilizes a Random Forest Regressor model for prediction and includes steps for data loading, preprocessing, model training, and evaluation.

## Dataset

The dataset used in this project is `books.csv`. It contains information about books, including average rating, number of pages, ratings count, text reviews count, and language code.

## Notebook Structure

The notebook is organized into several sections:

1.  **Data Loading and Initial Exploration:** Loading the `books.csv` file and performing initial checks on the data, such as viewing column names and basic information.
2.  **Data Preprocessing:**
    *   Selecting relevant features for the model.
    *   Handling missing values by dropping rows with `NaN` in selected columns.
    *   Encoding the categorical feature `language_code` using Label Encoding.
3.  **Exploratory Data Analysis (EDA):** Visualizations to understand the distributions of key features and their relationships with the target variable (`average_rating`). This includes:
    *   Histogram of average ratings.
    *   Scatter plots showing the relationship between average rating and number of pages, and average rating and ratings count.
    *   Count plot for language codes.
    *   Box plot showing the distribution of average ratings by language.
4.  **Model Training:**
    *   Splitting the data into training and testing sets.
    *   Training a Random Forest Regressor model on the training data.
5.  **Model Evaluation:**
    *   Predicting average ratings on the test set.
    *   Evaluating the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
6.  **Model Interpretation:**
    *   Visualizing feature importances to understand which features contribute most to the predictions.
    *   Creating a scatter plot of actual vs. predicted ratings to visualize the model's performance.
    *   Generating a residual plot to examine the errors in predictions.

## Requirements

To run this notebook, you will need the following Python libraries:

*   pandas
*   scikit-learn
*   matplotlib
*   seaborn
*   IPython (typically included in environments like Colab or Jupyter)

These can be installed using pip:
## How to Run

1.  Clone the repository or download the notebook file.
2.  Make sure you have the `books.csv` dataset in the same directory as the notebook or provide the correct path to the file.
3.  Open the notebook in Google Colab, Jupyter Notebook, or JupyterLab.
4.  Run the cells sequentially.

## Results

The notebook outputs the MAE, MSE, and R² values, which indicate the model's performance on the test set. The visualizations provide further insights into the data and model behavior.

## Future Improvements

*   Experiment with different regression models (e.g., Gradient Boosting, Support Vector Regression).
*   Perform more extensive feature engineering.
*   Tune model hyperparameters for better performance.
*   Explore different data cleaning strategies.
