Istanbul Rent Prediction

Features

Data Collection: Merges datasets from multiple CSV files containing rental information.

Data Cleaning: Handles missing values, converts categorical variables, and standardizes numerical data.

Feature Engineering: Extracts and encodes relevant features such as area, rooms, town, and year.

Machine Learning Model: Trains a Random Forest Regressor to predict rental prices.

Future Predictions: Generates predictions for rental prices between 2022 and 2025.

Visualization: Includes graphical representations of prediction results and error distributions.

Repository Structure

IstanbulRentPrediction/
├── dataset/
│   ├── 5_9_2022_sahibinden_ev.csv
│   ├── 22_5_2022_sahibinden_ev.csv
│   ├── 26_5_2022_sahibinden_ev.csv
├── main.py          # Main script for data processing and model training
├── requirements.txt # Required Python libraries
├── README.md        # Project documentation

Getting Started

Prerequisites

Python 3.8 or higher

Required libraries (see requirements.txt):

pandas

numpy

scikit-learn

matplotlib

re

Installation

Clone the repository:

git clone https://github.com/yagizterzi/IstanbulRentPrediction.git
cd IstanbulRentPrediction

Install the required libraries:

pip install -r requirements.txt

Running the Project

Place your data files in the dataset/ directory.

Run the main.py script:

python main.py

View the predictions and visualizations in the console and output plots.

Project Workflow

Data Loading:

Merges datasets from the dataset/ folder.

Extracts the date from filenames and adds it as a feature.

Data Cleaning:

Handles missing or invalid values.

Converts text fields (e.g., price, area) into numeric types.

Feature Engineering:

Encodes categorical variables (e.g., town) using one-hot encoding.

Extracts year from the date.

Model Training:

Splits the data into training and testing sets.

Trains a Random Forest Regressor.

Evaluation:

Evaluates the model using metrics like MSE, RMSE, MAE, and R².

Visualization:

Plots prediction results and error distributions.

Results

The model predicts rental prices based on features such as area, number of rooms, location, and year.

Visualizations provide insights into model performance and future price trends.

Future Work

Expand the dataset with more recent and diverse data.

Experiment with other regression models like Gradient Boosting or Neural Networks.

Include additional features such as proximity to public transport or amenities.

Deploy the model as a web app for real-time predictions.

Contributions

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the project.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

Author: Yağız Terzi

GitHub: github.com/yagizterzi

Email: yagizterzi198@gmail.com
This project is licensed under the MIT License. See the LICENSE file for details.

Contact

Created by Yağız Terzi - feel free to reach out for questions or collaboration!
