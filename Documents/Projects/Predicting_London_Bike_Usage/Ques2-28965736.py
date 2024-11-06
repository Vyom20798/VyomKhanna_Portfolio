"""
Importing Python Libraries:
    pandas: Data Handling, 
    seaborn and matplotlib: creating charts, 
    numpy: numerical operations),
    scikit-learn: machine learning
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


"""
pd.read_csv asking python to load the file
This file contains data of 3 months 
from 1st July 2023 to 30th September 2023
Start Station: Hyde Park Corner, Hyde Park
"""
weather_df = pd.read_csv('London 2023-07-01 to 2023-09-30.csv')

"""
Using pd.to_datetime converting datetime column into
a format that is easier to work with.
"""
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

"""
Extracting and creating new column:
    .dt.day_name() will give name of the day (eg: Monday, Tuesday)
    .dt.month will give month number (eg: 7,8 or 9)
    .dt.dayofweek will give day number with Monday = 0 and Sunday = 6
    .dt.isocalender().week will give week number
    .isin([5, 6]).astype(int) will tell if the day is weekend or not
"""
weather_df['day'] = weather_df['datetime'].dt.day_name()
weather_df['Month'] = weather_df['datetime'].dt.month
weather_df['day_of_week'] = weather_df['datetime'].dt.dayofweek
weather_df['weekend'] = weather_df['day_of_week'].isin([5, 6]).astype(int)
weather_df['week'] = weather_df['datetime'].dt.isocalendar().week


"""
Using .pd.to_datetime on column sunrise and sunset to make into 
a simpler format
"""
weather_df['sunrise'] = pd.to_datetime(weather_df['sunrise']).dt.time
weather_df['sunset'] = pd.to_datetime(weather_df['sunset']).dt.time

"""
Using .drop(columns=[Column Names]) to drop those columns 
which are completely blank
"""
weather_df = weather_df.drop(columns=['severerisk', 'snow', 'snowdepth', 'stations'])

"""
using pd.get_dummies as it converts certain types of data
into a format that can be used in calculations.
"""
weather_df = pd.get_dummies(weather_df, columns=['day', 'Month', 'preciptype'], drop_first=True)

"""
Making a list having code picks out 
specific pieces of information (like temperature and humidity) 
that are important for further analysis.
"""
features = [
    'tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'dew', 'humidity',
    'precip', 'precipprob', 'precipcover', 'windspeed', 'winddir', 'cloudcover', 'solarradiation',
    'solarenergy', 'uvindex', 'day_of_week', 'weekend'
]


final_data = weather_df[features]
print("Prepared Bike Usage Features:")
print(final_data.head())

"""
Summarising weekly data, 
calculating sum and mean for the data
"""
weekly_agg = weather_df.groupby('week').agg({
    'tempmax': 'mean',
    'tempmin': 'mean',
    'temp': 'mean',
    'feelslikemax': 'mean',
    'feelslikemin': 'mean',
    'feelslike': 'mean',
    'dew': 'mean',
    'humidity': 'mean',
    'precip': 'sum',
    'precipprob': 'mean',
    'precipcover': 'mean',
    'windspeed': 'mean',
    'winddir': 'mean',
    'cloudcover': 'mean',
    'solarradiation': 'mean',
    'solarenergy': 'mean',
    'uvindex': 'mean'
}).reset_index()


"""
Calculation new features (Feature Engineering) from the existing data:
    1. temp_feelslike = temp * feelslike
    2. humidity_temp = humidity * temp
    3. wind_humidity = windspeed * humidity
"""
weather_df['temp_feelslike'] = weather_df['temp'] * weather_df['feelslike']
weather_df['humidity_temp'] = weather_df['humidity'] * weather_df['temp']
weather_df['wind_humidity'] = weather_df['windspeed'] * weather_df['humidity']


"""
Creating lagged feature for temp, humidity and precip
This is done for 3 days
Lagged feature means it will compare the data for the 
aforementioned column for 4 dates.
Thus helping to understand impact of weather the previous day on 
current conditions.
"""
for lag in range(1, 4):
    weather_df['temp_lag_' + str(lag)] = weather_df['temp'].shift(lag)
    weather_df['humidity_lag_' + str(lag)] = weather_df['humidity'].shift(lag)
    weather_df['precip_lag_' + str(lag)] = weather_df['precip'].shift(lag)

"""
Creating a roll over matrix of:
    Temp: mean
    Humidity: mean
    precip: sum
to understand the among difference time ranges
that is 7 days, 14 days and 30 days
"""
for roll_data in [7, 14, 30]:
    weather_df['temp_roll_mean_' + str(roll_data)] = weather_df['temp'].rolling(roll_data).mean()
    weather_df['humidity_roll_mean_' + str(roll_data)] = weather_df['humidity'].rolling(roll_data).mean()
    weather_df['precip_roll_sum_' + str(roll_data)] = weather_df['precip'].rolling(roll_data).sum()


"""
left joining all data into 1 using .merge on week
"""
# Combine all features
final_data = weather_df[
    features +
    ['temp_feelslike', 'humidity_temp', 'wind_humidity'] +
    ['temp_lag_' + str(lag) for lag in range(1, 4)] +
    ['humidity_lag_' + str(lag) for lag in range(1, 4)] +
    ['precip_lag_' + str(lag) for lag in range(1, 4)] +
    ['temp_roll_mean_' + str(roll_data) for roll_data in [7, 14, 30]] +
    ['humidity_roll_mean_' + str(roll_data) for roll_data in [7, 14, 30]] +
    ['precip_roll_sum_' + str(roll_data) for roll_data in [7, 14, 30]]
]
# Merge weekly aggregates back into the original dataset
weather_df = weather_df.merge(weekly_agg, on='week', how='left')

"""
# Normalising the numerical variables
"""
numerical_features = final_data.select_dtypes(include=[np.number]).columns.tolist()
scaler = MinMaxScaler()
final_data[numerical_features] = scaler.fit_transform(final_data[numerical_features])


"""Plotting the correlation matrix of final_data()"""
plt.figure(figsize=(50, 35))
correlation_matrix = final_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': .5})
plt.title("Correlation Heatmap of Engineered Features")
plt.show()


"""
Importing SimpleInputer from sklearn.impute
This is imported so that we can replace the missing value (NaN)
with the average value of whole column, thus 
ensuring the data is complete and ready for analysis.
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
final_data = pd.DataFrame(imputer.fit_transform(final_data), columns=final_data.columns)


"""
Importing train_test_split from sklearn.model_selection
This code separates the data into features (x_temp) 
and the target variable (y_temp), where features include 
all columns except 'temp', and the target variable is the 
'temp' column itself.
Since my target is to create a model on temperature basis 
I will be doing my test on temp feature. For other feature
we can just change the feature name
"""
from sklearn.model_selection import train_test_split
x_temp = final_data.drop(columns=['temp'])  
y_temp = final_data['temp']  


"""
Split the data into training and testing sets
Testing on 25% data that is test_size = 0.25
And random state of 40, meaning if the data runs on other day
you will get the exact same training and testing sets, 
making it easier to compare results and debug.
"""
x_temp_train, x_temp_test, y_temp_train, y_temp_test = train_test_split(x_temp, y_temp, test_size=0.25, random_state=40)

"""
Importing mean_squared_error, r2_score ,mean_absolute_error from 
sklearn.metrics to do calculate 'R square mean', 'mean squared error'
and 'mean_absolute_error'. Also using hyper parameters

List of Models:
    Linear Regression: A simple linear model that predicts the target variable by fitting a straight line to the data.
    Decision Tree: A model that splits the data into branches to make predictions.
    Random Forest: An ensemble of decision trees that improves prediction accuracy by averaging their results.
    Neural Network: A model inspired by the human brain, consisting of layers of interconnected nodes.
    SVM (Support Vector Machine): A model that finds the best boundary (or hyperplane) to separate data points.
    KNN (K-Nearest Neighbors): A model that predicts the target variable based on the values of the nearest neighbors.
    XGBoost: A powerful model that uses gradient boosting to improve prediction accuracy.
"""
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error

"""
Common Terminology:
    mse: mean squared error
    r2: r2 score
    mae: mean absolute error
    pred: prediction
"""
"""
Creating a model for linear regression
Importing LinearRegression from sklearn.linear_model for testing linear regression
"""
from sklearn.linear_model import LinearRegression
linear_regression_model = LinearRegression()
linear_regression_model.fit(x_temp_train, y_temp_train)
y_pred_linear_regression = linear_regression_model.predict(x_temp_test)
mse_linear_regression = mean_squared_error(y_temp_test, y_pred_linear_regression)
r2_linear_regression = r2_score(y_temp_test, y_pred_linear_regression)
mae_linear_regression = mean_absolute_error(y_temp_test, y_pred_linear_regression)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_temp_test, y=y_pred_linear_regression)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--') 
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Linear Regression')
plt.tight_layout()
plt.show()

"""
print('Linear Regression')
print('MSE:',round(mse_linear_regression,4),
      'R2:',round(r2_linear_regression,4),
      'MAE:',round(mae_linear_regression,4))
"""

"""
Creating a model for Decision Tree
Importing DecisionTreeRegressor from sklearn.tree
"""
from sklearn.tree import DecisionTreeRegressor
decision_tree_model = DecisionTreeRegressor(random_state=48)
decision_tree_model.fit(x_temp_train, y_temp_train)
y_pred_decision_tree = decision_tree_model.predict(x_temp_test)
mse_decision_tree = mean_squared_error(y_temp_test, y_pred_decision_tree)
r2_decision_tree = r2_score(y_temp_test, y_pred_decision_tree)
mae_decision_tree = mean_absolute_error(y_temp_test, y_pred_decision_tree)

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_temp_test, y=y_pred_decision_tree)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Decision Tree')
plt.tight_layout()
plt.show()
"""
print('Decision Tree')
print('MSE:',round(mse_decision_tree,4), 'R2:',
      round(r2_decision_tree,4),'MAE:',round(mae_decision_tree,4))
"""

"""
Creating a model for Random forest
Importing RandomForestRegressor from sklearn.ensemble
"""
from sklearn.ensemble import RandomForestRegressor
Random_forest_model = RandomForestRegressor(random_state=48)
Random_forest_model.fit(x_temp_train, y_temp_train)
y_pred_Random_forest = Random_forest_model.predict(x_temp_test)
mse_Random_forest = mean_squared_error(y_temp_test, y_pred_Random_forest)
r2_Random_forest = r2_score(y_temp_test, y_pred_Random_forest)
mae_Random_forest = mean_absolute_error(y_temp_test, y_pred_Random_forest)

plt.subplot(1, 3, 3)
sns.scatterplot(x=y_temp_test, y=y_pred_Random_forest)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Random Forest')
plt.tight_layout()
plt.show()
"""
print('Random Forest')
print('MSE:',round(mse_Random_forest,4),'R2:',round(r2_Random_forest,4),
      'MAE:',round(mae_Random_forest,4))
"""

"""
Creating a model for neural networks
Importing MLPRegressor from sklearn.neural_network
"""
from sklearn.neural_network import MLPRegressor
neural_network_model = MLPRegressor(random_state=48, max_iter=100)
neural_network_model.fit(x_temp_train, y_temp_train)
y_pred_neural_network = neural_network_model.predict(x_temp_test)
mse_neural_network = mean_squared_error(y_temp_test, y_pred_neural_network)
r2_neural_network = r2_score(y_temp_test, y_pred_neural_network)
mae_neural_network = mean_absolute_error(y_temp_test, y_pred_neural_network)

plt.scatter(y_temp_test, y_pred_neural_network)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for Neural Network')
plt.tight_layout()
plt.show()
"""
print('Neural Network')
print('MSE:',round(mse_neural_network,4),'R2:',round(r2_neural_network,4),
      'MAE:',round(mae_neural_network,4))
"""

"""
Creating a model for XGBoost
Importing xgboost
"""
import xgboost as xgb
XGBoost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=48)
XGBoost_model.fit(x_temp_train, y_temp_train)
y_pred_XGBoost = XGBoost_model.predict(x_temp_test)
mse_XGBoost = mean_squared_error(y_temp_test, y_pred_XGBoost)
r2_XGBoost = r2_score(y_temp_test, y_pred_XGBoost)
mae_XGBoost = mean_absolute_error(y_temp_test, y_pred_XGBoost)

plt.scatter(y_temp_test, y_pred_XGBoost)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for XGBoost')
plt.tight_layout()
plt.show()
"""
print('XGBoost')
print('MSE:',round(mse_XGBoost,4),'R2:',round(r2_XGBoost,4),
      'MAE:',round(mae_XGBoost,4))
"""

"""
Creating a model for SVM
Importing svr from sklearn.svm for support vector machine
"""
from sklearn.svm import SVR
svm_model = SVR(kernel='rbf')
svm_model.fit(x_temp_train, y_temp_train)
y_temp_pred_svm = svm_model.predict(x_temp_test)
mse_svm = mean_squared_error(y_temp_test, y_temp_pred_svm)
r2_svm = r2_score(y_temp_test, y_temp_pred_svm)
mae_svm = mean_absolute_error(y_temp_test, y_temp_pred_svm)

plt.scatter(y_temp_test, y_temp_pred_svm)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for SVM')
plt.tight_layout()
plt.show()
"""
print('SVM')
print('MSE:',round(mse_svm,4),'R2:',round(r2_svm,4),
      'MAE:',round(mae_svm,4))
"""

"""
Creating a model for KNN
Importing KNeighborsRegressor from sklearn.neighbors
n_neighbors=5 means the model will look at the 
5 closest data points (neighbors) to make a prediction
"""
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(x_temp_train, y_temp_train)
y_temp_pred_knn = knn_model.predict(x_temp_test)
mse_knn = mean_squared_error(y_temp_test, y_temp_pred_knn)
r2_knn = r2_score(y_temp_test, y_temp_pred_knn)
mae_knn = mean_absolute_error(y_temp_test, y_temp_pred_knn)

plt.scatter(y_temp_test, y_temp_pred_knn)
plt.plot([min(y_temp_test), max(y_temp_test)], [min(y_temp_test), max(y_temp_test)], color='red', linestyle='--')  # Perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values for KNN')
plt.tight_layout()
plt.show()
"""
print('KNN')
print('MSE:',round(mse_knn,4),'R2:',round(r2_knn,4),
      'MAE:',round(mae_knn,4))
"""


models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=48),
    'Random Forest': RandomForestRegressor(random_state=48),
    'Neural Network': MLPRegressor(random_state=48, max_iter=100),
    'SVM': SVR(kernel='rbf'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=48)
}

"""
Creating an empty dictionary named 'results' to 
store the calculated mse, r2 and mae of the models
"""
results = {}

"""
Trainng and evalutaing the 7 models
"""
for name, model in models.items():
    model.fit(x_temp_train, y_temp_train)
    y_temp_pred = model.predict(x_temp_test)
    mse = mean_squared_error(y_temp_test, y_temp_pred)
    r2 = r2_score(y_temp_test, y_temp_pred)
    mae = mean_absolute_error(y_temp_test, y_temp_pred)
    results[name] = {'MSE': mse, 'R2': r2, 'MAE': mae}

"""
Printing the results
"""
for name, metrics in results.items():
    print(f"{name} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}")

"""
Finding the best model using minimum MAE that is mean absolute error
"""
best_model = min(results, key=lambda k: results[k]['MAE'])
print(f'\nBest Model based on MAE: {best_model}')