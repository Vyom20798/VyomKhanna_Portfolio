"""
Importing Pythom Libraries:
    pandas used for data manipulation and analysis.
    matplotlib.pyplot and seaborn used for creating visualisations
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



"""-------- CSV READING, CONCATENATION AND FILTERING """
"""
Using pd.read_csv reading all the .csv files
My Studnet ID is 28965736, hence working on data of June. 
pd.read_csv(): Reads each CSV file.
"""
df1 = pd.read_csv('372JourneyDataExtract29May2023-04Jun2023.csv')
df2 = pd.read_csv('374JourneyDataExtract12Jun2023-18Jun2023.csv')
df3 = pd.read_csv('373JourneyDataExtract05Jun2023-11Jun2023.csv')
df4 = pd.read_csv('375JourneyDataExtract19Jun2023-30Jun2023.csv')


"""
Using pd.concat function here to make it 1 .csv file, 
using ignore_index = True meaning all columns will come one 
under the other
pd.concat(dataframe names,ignore_index=True): Concatenates all data frames 
into one single data frame, resetting the index.
"""
df = pd.concat([df1, df2, df3, df4], ignore_index=True)


"""
My data has some dates of Month of May hence filtering out 
data and keeping data for month of June
pd.to_datetime(): Converts the 'Start date' 
column to datetime format.
"""
df['Start date'] = pd.to_datetime(df['Start date'], 
                                  format='%Y-%m-%d %H:%M')
june_df = df[df['Start date'].dt.month == 6]


"""
Filtering the Dataframe to remove Total Duration (ms) 
where time is less than 1 Minute or More than 1 Day
1 Min = 60000 ms and 1 Day = 86400000 ms
"""
june_df = june_df[(june_df['Total duration (ms)'] >= 60000) 
                  & (june_df['Total duration (ms)'] <= 86400000)]


"""-------- COLUMN ADDITION"""
"""Creating a column 'day_of_the_week' using .dt.day_name()
which will show the day of that date"""
june_df['day_of_the_week'] = june_df['Start date'].dt.day_name()

"""Creating a column 'start date temp' using .dt.date
in start date which will show the date in YYYY-MM-DD Format"""
june_df['Start date temp'] = june_df['Start date'].dt.date


"""Bifurcation time into Morning/Afternoon/Evening/Night"""
def time_of_day(Time):
    if 6 <= Time < 12:
        return 'Morning'
    elif 12 <= Time < 18:
        return 'Afternoon'
    elif 18 <= Time < 23:
        return 'Evening'
    else:
        return 'Night'

"""
Creating a column 'time_of_day' using .dt.hour.apply(function)
on start date
"""
june_df['time_of_day'] = june_df['Start date'].dt.hour.apply(time_of_day)

"""
Merging Start Station and End Station and adding them as a new column
names as Route
"""
june_df['Route'] = june_df['Start station'] + ' - ' + june_df['End station']

"""Creating a new column 'hour' which will extract hour from 
start date column""" 
june_df['hour'] = june_df['Start date'].dt.hour


"""Creating a new column 'Time Interval'
Using concept of bins and label creating a time interval bucket"""
ms = [0, 5*60*1000, 15*60*1000, 30*60*1000, 60*60*1000, 24*60*60*1000]
intervals = ['0-5 Min', '5-15 Min', '15-30 Min', '30-60 Min', '60 Min - 1 Day']
june_df['Time Interval'] = pd.cut(june_df['Total duration (ms)'], bins=ms, labels=intervals)


"""Creating a fuction for a column of Week Number that will divide tag the date 
   as Week 1, Week 2, Week 3 and Week 4"""
def Week_Number(date):
    if date >= pd.Timestamp('2023-06-01') and date < pd.Timestamp('2023-06-08'):
        return 'Week 1'
    elif date >= pd.Timestamp('2023-06-08') and date < pd.Timestamp('2023-06-15'):
        return 'Week 2'
    elif date >= pd.Timestamp('2023-06-15') and date < pd.Timestamp('2023-06-22'):
        return 'Week 3'
    else:
        return 'Week 4'

"""Adding the above mentioned in fuction in the dataframe"""
june_df['Week_Number'] = june_df['Start date'].apply(Week_Number)

"""-------- DROPPING UNWATED DATAFRAMES using del function"""
del df, df1, df2, df3, df4

"""--------  FINDING INSGHITS and PLOTS"""

"""
For Plotting
    Type of chart: Pie Chart, Bar Chart, Horizontal bar chart
    Size: figure size will be (x,y)
        x Means Width
        y Means Height
    Autopct means upto 2 decimal points
    legend will then help identify which color corresponds 
    to which bike model
    Tight layout will not let one plot overlap over the other
"""

"""
--------  1. Bike Model Frequency
Calculating Bike Model Frequency of Classic and Psbc_ebike 
using .value_count()
"""
rental_frequency = june_df['Bike model'].value_counts()
bike_frequence = june_df['Bike model'].value_counts().sum()
print('Number of bike rented in June:')
print(bike_frequence)
print('Bike Model frequency for June is:')
print(rental_frequency)

""" 
Creating a pie chart for Bike Model Frequency     
"""
plt.figure(figsize=(10, 5))
plt.title('Bike Models Rental Frequency')
plt.pie(rental_frequency, labels=rental_frequency.index, autopct='%1.2f%%', startangle=60, shadow = 'True')
plt.legend(rental_frequency.index, title="Bike Models")
plt.tight_layout()
plt.show()

"""
--------  2. Time of the day Frequency
Calculating bike frequency of during Morniing, Afternoon, Evening and Night
using .value_count()
"""
time_route = june_df['time_of_day'].value_counts()
print('Frequency during different time of the day:')
print(time_route)


"""
Creating a pie chart for Bike Model Frequency 
"""
plt.figure(figsize=(10, 5))
plt.title('Time of the day Rental Frequency')
plt.pie(time_route, labels=time_route.index, autopct='%1.2f%%', startangle=60, shadow = 'True')
plt.legend(time_route.index, title="Bike Models")
plt.tight_layout()
plt.show()


"""
--------  3. Number of Trips every day
Calculating bike frequency of every date
using .groupby() and .size()
"""
trips_per_day = june_df.groupby('Start date temp').size()
print('Trips per day are:')
print(trips_per_day)

"""
Creating a bar chart showing number of trips everyday in June
Here: 
    X-axis shows start date 
    Y-Axis shows number of trips
"""
plt.figure(figsize=(10, 5))
plt.plot(trips_per_day.index, trips_per_day.values, marker='o', linestyle='-', color= 'red')
plt.bar(trips_per_day.index,trips_per_day.values,color='blue')
plt.title('Monthly Rental Frequency')
plt.xlabel('Start date')
plt.ylabel('Number of Trips')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


"""
--------  4. Frequent Start Station
Calculating top 10 start stations using .head(10)
"""
frequent_start_station = june_df['Start station'].value_counts().head(10)
print('Top 10 Start Station in June are:')
print(frequent_start_station)

"""
Creating a horizontal bar showing top 10 start stations
Here:
    X-Axis shows rental frequency
    Y-Axis shows Top 10 Start station
"""
frequent_start_station = frequent_start_station.sort_values(ascending=True)
plt.figure(figsize=(10, 5))
plt.barh(frequent_start_station.index,frequent_start_station.values,color = 'Yellow')
plt.xlabel('Frequency')
plt.ylabel('Start station')
plt.xticks(rotation=45)
plt.title('Top 10 Start Station in June')
plt.tight_layout()
plt.legend()
plt.show()


"""
-------- 5. Frequent End Station
Calculating top 10 end stations using .head(10)
"""
frequent_end_station = june_df['End station'].value_counts().head(10)
print('Top 10 End Station in June are:')
print(frequent_end_station)

"""
Creating a horizontal bar showing top 10 end stations
Here:
    X-Axis shows rental frequency
    Y-Axis shows Top 10 end stations
"""
frequent_end_station = frequent_end_station.sort_values(ascending=True)
plt.figure(figsize=(10, 5))
plt.barh(frequent_end_station.index,frequent_end_station.values,color = 'Green')
plt.xlabel('Frequency')
plt.ylabel('End Station')
plt.xticks(rotation=45)
plt.title('Top 10 End station in June')
plt.tight_layout()
plt.legend()
plt.show()


"""
--------  6. Frequent Route
Calculating top 10 routes using .head(10)
"""
frequent_route = june_df['Route'].value_counts().head(10)
print('Top 10 Routes in June are:')
print(frequent_route)


"""
Importing textwrap library so make the Y-Axis(Route Name) more readable 
and presentable
"""
frequent_route = frequent_route.sort_values(ascending=True)
import textwrap
labels = [textwrap.fill(label, width=30) for label in frequent_route.index]

"""
Creating a horizontal bar for Top 10 routes
Here:
    X-Axis shows Rental Frequency
    Y-Axis shows Top 10 Routes
"""
plt.figure(figsize=(12, 8))
plt.barh(frequent_route.index,frequent_route.values,color = 'Blue')
plt.xlabel('Frequency')
plt.ylabel('Route')
plt.title('Top 10 Routes in June')
plt.yticks(ticks=range(len(labels)), labels=labels)
plt.tight_layout()
plt.legend()
plt.show()


"""
--------  7. Hour Rental
Calculating bike frequent by hour for whole month
"""
hour_rental = june_df.groupby('hour').size()

"""
Calculating bike frequent by hour and date for whole month
"""
daily_hour_rental = june_df.groupby(['hour', 'Start date temp']).size()

"""
Creating a scatterplot showing hour rental for the whole month
Here:
    X-Axis: Hours of the day
    Y-Axis: Hour Rental for the whole month
"""
plt.figure(figsize=(10, 5))
plt.plot(hour_rental.index, hour_rental.values, marker='o', linestyle='-')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Rentals')
plt.title('Number of Rentals per Hour in June')
plt.grid(True)
plt.tight_layout()
plt.show()


"""
--------  8.Frequqent Bike Number
Converting Bike Number into string format for easy analysis
Using .value_counts() and .head(10) calculation 
Top 10 bike number used for the month
"""
june_df['Bike number'] = june_df['Bike number'].astype(str)
frequent_bike = june_df['Bike number'].value_counts().head(10)
print('Top 10 Bike Number used in June are:')
print(frequent_bike)

"""
Creating a bar chart showing rental frequency of bar chart
Here:
    X-Axis: Bike numbers
    Y-Axis: Bike Number Rental Frequency
"""
plt.figure(figsize=(10, 5))
plt.bar(frequent_bike.index,frequent_bike.values)
plt.xlabel('Bike Number')
plt.ylabel('Bike Number rental frequency')
plt.title('Top 10 Bike Number in June')
plt.tight_layout()
plt.legend()
plt.show()


"""--------  9. Weekwise Insights """
""" 
Doing the above analysis that is:
    1. Total Duration (ms) total
    2. Total Duration (ms) mean
    3. Bike_Model frequency
    4. Day rental frequency
    5. Top 10 Routes
    6. Top 10 Start Stations
    7. Top 10 End Stations
    8. Top 10 Bike Number
    9. Daily hour rental
weekwise
"""

"""
--------  9.1 Week 1
Week 1 is start date temp from 2023-06-01 to 2023-06-07
Creating a variable Week1 using .query() which will filter Week1 data
from the Week_number column
"""
Week1 = june_df.query('Week_Number == "Week 1"')
Week1_total = round(Week1['Total duration (ms)'].sum() / 3600000,2) 
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week1_mean = round(Week1['Total duration (ms)'].mean() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week1_rental_frequence = Week1['Bike model'].value_counts()
Week1_day_frequency = Week1['day_of_the_week'].value_counts()
Week1_top10_route = Week1['Route'].value_counts().head(10)
Week1_top10_start_stations = Week1['Start station'].value_counts().head(10)
Week1_top10_end_stations = Week1['End station'].value_counts().head(10)
Week1_daily_hour_rental = Week1.groupby(['hour', 'Start date temp']).size()
Week1_top10_bike = Week1['Bike number'].value_counts().head(10)


"""
--------  9.2 Week 2
Week 2 is start date temp from 2023-06-08 to 2023-06-14
"""
Week2 = june_df.query('Week_Number == "Week 2"')
Week2_total = round(Week2['Total duration (ms)'].sum() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week2_mean = round(Week2['Total duration (ms)'].mean() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week2_rental_frequence = Week2['Bike model'].value_counts()
Week2_day_frequency = Week2['day_of_the_week'].value_counts()
Week2_top10_route = Week2['Route'].value_counts().head(10)
Week2_top10_start_stations = Week2['Start station'].value_counts().head(10)
Week2_top10_end_stations = Week2['End station'].value_counts().head(10)
Week2_daily_hour_rental = Week2.groupby(['hour', 'Start date temp']).size()
Week2_top10_bike = Week2['Bike number'].value_counts().head(10)

"""
--------  9.3 Week 3
Week 3 is start date temp from 2023-06-15 to 2023-06-22"""
Week3 = june_df.query('Week_Number == "Week 3"')
Week3_total = round(Week3['Total duration (ms)'].sum() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week3_mean = round(Week3['Total duration (ms)'].mean() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week3_rental_frequence = Week3['Bike model'].value_counts()
Week3_day_frequency = Week3['day_of_the_week'].value_counts()
Week3_top10_route = Week3['Route'].value_counts().head(10)
Week3_top10_start_stations = Week3['Start station'].value_counts().head(10)
Week3_top10_end_stations = Week3['End station'].value_counts().head(10)
Week3_daily_hour_rental = Week3.groupby(['hour', 'Start date temp']).size()
Week3_top10_bike = Week3['Bike number'].value_counts().head(10)

"""
--------  9.4 Week 4
Week 4 is start date temp from 2023-06-23 to 2023-06-30
"""
Week4 = june_df.query('Week_Number == "Week 4"')
Week4_total = round(Week4['Total duration (ms)'].sum() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week4_mean = round(Week4['Total duration (ms)'].mean() / 3600000,2)
"""Dividing by 3600000 so that it miliseconds 
could be converted into hours"""
Week4_rental_frequence = Week4['Bike model'].value_counts()
Week4_day_frequency = Week4['day_of_the_week'].value_counts()
Week4_top10_route = Week4['Route'].value_counts().head(10)
Week4_top10_start_stations = Week4['Start station'].value_counts().head(10)
Week4_top10_end_stations = Week4['End station'].value_counts().head(10)
Week4_daily_hour_rental = Week4.groupby(['hour', 'Start date temp']).size()
Week4_top10_bike = Week4['Bike number'].value_counts().head(10)

"""--------  9.5 Weekends """
weekend = june_df.query('`day_of_the_week` == "Saturday" or `day_of_the_week` == "Sunday"')
weekend_total = weekend['Total duration (ms)'].sum() / 3600000
weekend_rental_frequency = weekend['Bike model'].value_counts()
weekend_frequent_route = weekend['Route'].value_counts().head(10)
weekend_top10_route = weekend['Route'].value_counts().head(10)
weekend_top10_start_stations = weekend['Start station'].value_counts().head(10)
weekend_top10_end_stations = weekend['End station'].value_counts().head(10)
weekend_top10_bike = weekend['Bike number'].value_counts().head(10)

"""--------  9.6 Weekdays """
weekday = june_df.query('`day_of_the_week` != "Saturday" and `day_of_the_week` != "Sunday"')
weekday_total = weekday['Total duration (ms)'].sum() / 3600000
weekday_rental_frequency = weekday['Bike model'].value_counts()
weekday_frequent_route = weekday['Route'].value_counts().head(10)
weekday_top10_route = weekday['Route'].value_counts().head(10)
weekday_top10_start_stations = weekday['Start station'].value_counts().head(10)
weekday_top10_end_stations = weekday['End station'].value_counts().head(10)
weekday_top10_bike = weekday['Bike number'].value_counts().head(10)


"""
Creating a pie chart showing Total Duration (ms) of bike weekwise
"""
Weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
plt.figure(figsize=(10, 5))
plt.pie([Week1_total,Week2_total,Week3_total,Week4_total], labels=Weeks, autopct='%1.2f%%', startangle=60)
plt.title('Total Duration (ms) for Each Week in June')
plt.legend(Weeks, title="Weeks")
plt.tight_layout()
plt.show()

"""
Creating a counterplot chart showing monthly rental frequency daywise
""" 
plt.figure(figsize=(10, 5))
sns.countplot(x='day_of_the_week', 
              data=june_df, 
              order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
"""
Defining an order in which the days has to be plotted
"""
plt.xlabel('Day')
plt.ylabel('Ride Frequency')
plt.title('Ride Counts by Day_of_the_Week in June')
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np

"""
Defining the order in which the Day of the week is needed
"""
day_of_the_week_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

"""
Updating the column in the dataframe
"""
june_df['day_of_the_week'] = pd.Categorical(june_df['day_of_the_week'], categories=day_of_the_week_order, ordered=True)

"""
Grouping by data on basis of Week_Number and day_of_the_week
The .size() function counts the number of rows in each group.
The .unstack() function reshapes the Series to make the 
day_of_the_week values into columns. 
categories: Gets the column names (days of the week) from the grouped data.
weeks: Gets the row indices (week numbers) from the grouped data.
bar_width: Sets the width of each bar in the chart.
index: Creates an array of indices corresponding to the categories (days of the week).
fig, ax = plt.subplots(figsize=(12, 7)): Creates a figure and axis object for the plot, setting the figure size.
bars = []: Initializes an empty list to store the bars.
for i, week in enumerate(weeks): Loops through each week to create the bars.
plt.bar(index + i * bar_width, grouped_data.loc[week], bar_width, label=week): Creates a set of bars for each week, 
shifting their positions horizontally to avoid overlap.
"""
grouped_data = june_df.groupby(['Week_Number', 'day_of_the_week']).size().unstack()
categories = grouped_data.columns
weeks = grouped_data.index
bar_width = 0.2
index = np.arange(len(categories))


fig, ax = plt.subplots(figsize=(12, 7))
bars = []
for i, week in enumerate(weeks):
    bars.append(plt.bar(index + i * bar_width, grouped_data.loc[week], bar_width, label=week))
ax.set_xlabel('Time of Day')
ax.set_ylabel('Number of Rentals')
ax.set_title('Number of Rentals per Day of the week for Each Week')
ax.set_xticks(index + bar_width * (len(weeks) - 1) / 2)
ax.set_xticklabels(categories)
ax.legend()

"""--------  10 Time of the day """
"""Calculating:
    1. Total duration (ms)
    2. bike_model frequency
    3. top 10 routes
time of the day wise
"""    
"""--------  10.1 Morning """
Morning = june_df.query('`time_of_day` == "Morning"')
Morning_total = Morning['Total duration (ms)'].sum() / 3600000
Morning_rental_frequency = Morning['Bike model'].value_counts()
Morning_frequent_route = Morning['Route'].value_counts().head(10)

"""--------  10.2 Afternoon """
Afternoon = june_df.query('`time_of_day` == "Afternoon"')
Afternoon_total = Afternoon['Total duration (ms)'].sum() / 3600000
Afternoon_rental_frequency = Afternoon['Bike model'].value_counts()
Afternoon_frequent_route = Afternoon['Route'].value_counts().head(10)

"""--------  10.3 Evening """
Evening = june_df.query('`time_of_day` == "Evening"')
Evening_total = Evening['Total duration (ms)'].sum() / 3600000
Evening_rental_frequency = Evening['Bike model'].value_counts()
Evening_frequent_route = Evening['Route'].value_counts().head(10)

"""--------  10.4 Night """
Night = june_df.query('`time_of_day` == "Night"')
Night_total = Night['Total duration (ms)'].sum() / 3600000
Night_rental_frequency = Night['Bike model'].value_counts()
Night_frequent_route = Night['Route'].value_counts().head(10)

time_of_day_order = ['Morning','Afternoon','Evening','Night']
june_df['time_of_day'] = pd.Categorical(june_df['time_of_day'], categories=time_of_day_order, ordered=True)

grouped_data = june_df.groupby(['Week_Number', 'time_of_day']).size().unstack().fillna(0)

categories = grouped_data.columns
weeks = grouped_data.index

bar_width = 0.2
index = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(12, 7))

bars = []
for i, week in enumerate(weeks):
    bars.append(plt.bar(index + i * bar_width, grouped_data.loc[week], bar_width, label=week))
ax.set_xlabel('Time of Day')
ax.set_ylabel('Number of Rentals')
ax.set_title('Number of Rentals per Time of Day for Each Week')
ax.set_xticks(index + bar_width * (len(weeks) - 1) / 2)
ax.set_xticklabels(categories)
ax.legend()

"""
--------  11. Correlation Matrix
Creating a correlation matrix of the june_df data
"""
plt.figure(figsize=(20, 15))
sns.heatmap(june_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


"""
Further EDA and Plotting could be done by bifurcating day_of_the_week and time_of_day
Calculatiing:
    1. Total Duration (ms)
    2. Bike_Model Frequency
    3. Top 10 Routes
"""

"""--------  12 Time of Day and Week """
"""--------  12.1 Morning Week 1 """
Morning_Week1 = june_df.query('time_of_day == "Morning" and Week_Number == "Week 1"')
Morning_Week1_Total = Morning_Week1['Total duration (ms)'].sum() / 3600000
Morning_Week1_rental_frequency = Morning_Week1['Bike model'].value_counts()
Morning_Week1_Route = Morning_Week1['Route'].value_counts().head(10)

"""--------  12.2 Afternoon Week 1 """
Afternoon_Week1 = june_df.query('time_of_day == "Afternoon" and Week_Number == "Week 1"')
Afternoon_Week1_Total = Afternoon_Week1['Total duration (ms)'].sum() / 3600000
Afternoon_Week1_rental_frequency = Afternoon_Week1['Bike model'].value_counts()
Afternoon_Week1_Route = Afternoon_Week1['Route'].value_counts().head(10)

"""--------  12.3 Evening Week 1 """
Evening_Week1 = june_df.query('time_of_day == "Evening" and Week_Number == "Week 1"')
Evening_Week1_Total = Evening_Week1['Total duration (ms)'].sum() / 3600000
Evening_Week1_rental_frequency = Evening_Week1['Bike model'].value_counts()
Evening_Week1_Route = Evening_Week1['Route'].value_counts().head(10)

"""--------  12.4 Night Week 1 """
Night_Week1 = june_df.query('time_of_day == "Night" and Week_Number == "Week 1"')
Night_Week1_Total = Night_Week1['Total duration (ms)'].sum() / 3600000
Night_Week1_rental_frequency = Night_Week1['Bike model'].value_counts()
Night_Week1_Route = Night_Week1['Route'].value_counts().head(10)

"""--------  12.5 Morning Week 2 """
Morning_Week2 = june_df.query('time_of_day == "Morning" and Week_Number == "Week 2"')
Morning_Week2_Total = Morning_Week2['Total duration (ms)'].sum() / 3600000
Morning_Week2_rental_frequency = Morning_Week2['Bike model'].value_counts()
Morning_Week2_Route = Morning_Week2['Route'].value_counts().head(10)

"""--------  12.6 Afternoon Week 2 """
Afternoon_Week2 = june_df.query('time_of_day == "Afternoon" and Week_Number == "Week 2"')
Afternoon_Week2_Total = Afternoon_Week2['Total duration (ms)'].sum() / 3600000
Afternoon_Week2_rental_frequency = Afternoon_Week2['Bike model'].value_counts()
Afternoon_Week2_Route = Afternoon_Week2['Route'].value_counts().head(10)

"""--------  12.7 Evening Week 2 """
Evening_Week2 = june_df.query('time_of_day == "Evening" and Week_Number == "Week 2"')
Evening_Week2_Total = Evening_Week2['Total duration (ms)'].sum() / 3600000
Evening_Week2_rental_frequency = Evening_Week2['Bike model'].value_counts()
Evening_Week2_Route = Evening_Week2['Route'].value_counts().head(10)

"""--------  12.8 Night Week 2 """
Night_Week2 = june_df.query('time_of_day == "Night" and Week_Number == "Week 2"')
Night_Week2_Total = Night_Week2['Total duration (ms)'].sum() / 3600000
Night_Week2_rental_frequency = Night_Week2['Bike model'].value_counts()
Night_Week2_Route = Night_Week2['Route'].value_counts().head(10)

"""--------  12.9 Morning Week 3 """
Morning_Week3 = june_df.query('time_of_day == "Morning" and Week_Number == "Week 3"')
Morning_Week3_Total = Morning_Week3['Total duration (ms)'].sum() / 3600000
Morning_Week3_rental_frequency = Morning_Week3['Bike model'].value_counts()
Morning_Week3_Route = Morning_Week3['Route'].value_counts().head(10)

"""--------  12.10 Afternoon Week 3 """
Afternoon_Week3 = june_df.query('time_of_day == "Afternoon" and Week_Number == "Week 3"')
Afternoon_Week3_Total = Afternoon_Week3['Total duration (ms)'].sum() / 3600000
Afternoon_Week3_rental_frequency = Afternoon_Week3['Bike model'].value_counts()
Afternoon_Week3_Route = Afternoon_Week3['Route'].value_counts().head(10)

"""--------  12.11 Evening Week 3 """
Evening_Week3 = june_df.query('time_of_day == "Evening" and Week_Number == "Week 3"')
Evening_Week3_Total = round(Evening_Week3['Total duration (ms)'].sum() / 3600000,2)
Evening_Week3_rental_frequency = Evening_Week3['Bike model'].value_counts()
Evening_Week3_Route = Evening_Week3['Route'].value_counts().head(10)

"""--------  12.12 Night Week 3 """
Night_Week3 = june_df.query('time_of_day == "Evening" and Week_Number == "Week 3"')
Night_Week3_Total = Night_Week3['Total duration (ms)'].sum() / 3600000
Night_Week3_rental_frequency = Night_Week3['Bike model'].value_counts()
Night_Week3_Route = Night_Week3['Route'].value_counts().head(10)

"""--------  12.13 Morning Week 4 """
Morning_Week4 = june_df.query('time_of_day == "Morning" and Week_Number == "Week 4"')
Morning_Week4_Total = Morning_Week3['Total duration (ms)'].sum() / 3600000
Morning_Week4_rental_frequency = Morning_Week3['Bike model'].value_counts()
Morning_Week4_Route = Morning_Week3['Route'].value_counts().head(10)

"""--------  12.14 Afternoon Week 4 """
Afternoon_Week4 = june_df.query('time_of_day == "Afternoon" and Week_Number == "Week 4"')
Afternoon_Week4_Total = Afternoon_Week3['Total duration (ms)'].sum() / 3600000
Afternoon_Week4_rental_frequency = Afternoon_Week3['Bike model'].value_counts()
Afternoon_Week4_Route = Afternoon_Week3['Route'].value_counts().head(10)

"""--------  12.15 Evening Week 4 """
Evening_Week4 = june_df.query('time_of_day == "Evening" and Week_Number == "Week 4"')
Evening_Week4_Total = Evening_Week3['Total duration (ms)'].sum() / 3600000
Evening_Week4_rental_frequency = Evening_Week3['Bike model'].value_counts()
Evening_Week4_Route = Evening_Week3['Route'].value_counts().head(10)

"""--------  12.16 Night Week 4 """
Night_Week4 = june_df.query('time_of_day == "Evening" and Week_Number == "Week 4"')
Night_Week4_Total = Night_Week3['Total duration (ms)'].sum() / 3600000
Night_Week4_rental_frequency = Night_Week3['Bike model'].value_counts()
Night_Week4_Route = Night_Week3['Route'].value_counts().head(10)