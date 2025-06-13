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
