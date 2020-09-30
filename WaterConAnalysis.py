import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib import interactive
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#read the water consumption dataset save it to dataframe
dataset = pd.read_csv('/Users/abeer/Downloads/202009-NEOMCamp_water.csv', index_col=1, parse_dates=True)
#extract additional data from datetime index
dataset['Year'] = dataset.index.year
dataset['Month'] = dataset.index.month
dataset['Weekday Name'] = dataset.index.weekday_name
dataset['Hour'] = dataset.index.hour
dataset['Day'] = dataset.index.day
#season column based on the month
conditions = [
    (dataset['Month'] == 8),
    (dataset['Month'] == 9)
    ]
# create a list of the values we want to assign for each condition
values = ['Summer', 'Fall']
# create a new column and use np.select to assign values to it using our lists as arguments
dataset['Season'] = np.select(conditions, values)
#Quarter of the day is it night or morning
condition = [
    (dataset['Hour'] <= 11),
    (dataset['Hour'] > 11)
    ]
# create a list of the values we want to assign for each condition
value = ['00 to 11:59 PM', '12 to 11:59 PM']
# create a new column and use np.select to assign values to it using our lists as arguments
dataset['Quarter'] = np.select(condition, value)
# add high temperature in °C based on the day time
conditionweather = [
    (dataset['Month'] == 8) & (dataset['Day'] == 16),
    (dataset['Month'] == 8) & (dataset['Day'] == 17),(dataset['Month'] == 8) & (dataset['Day'] == 18),(dataset['Month'] == 8) & (dataset['Day'] == 19),
    (dataset['Month'] == 8) & (dataset['Day'] == 20),(dataset['Month'] == 8) & (dataset['Day'] == 21),(dataset['Month'] == 8) & (dataset['Day'] == 22),
    (dataset['Month'] == 8) & (dataset['Day'] == 23),(dataset['Month'] == 8) & (dataset['Day'] == 24),(dataset['Month'] == 8) & (dataset['Day'] == 25),
    (dataset['Month'] == 8) & (dataset['Day'] == 26),(dataset['Month'] == 8) & (dataset['Day'] == 27),(dataset['Month'] == 8) & (dataset['Day'] == 28),
    (dataset['Month'] == 8) & (dataset['Day'] == 29),(dataset['Month'] == 8) & (dataset['Day'] == 30),(dataset['Month'] == 8) & (dataset['Day'] == 31),
    (dataset['Month'] == 9) & (dataset['Day'] == 1),(dataset['Month'] == 9) & (dataset['Day'] == 2),(dataset['Month'] == 9) & (dataset['Day'] == 3),
    (dataset['Month'] == 9) & (dataset['Day'] == 4),(dataset['Month'] == 9) & (dataset['Day'] == 5),(dataset['Month'] == 9) & (dataset['Day'] == 6),
    (dataset['Month'] == 9) & (dataset['Day'] == 7),(dataset['Month'] == 9) & (dataset['Day'] == 8),(dataset['Month'] == 9) & (dataset['Day'] == 9),
    (dataset['Month'] == 9) & (dataset['Day'] == 10),(dataset['Month'] == 9) & (dataset['Day'] == 11),(dataset['Month'] == 9) & (dataset['Day'] == 12),
    (dataset['Month'] == 9) & (dataset['Day'] == 13),(dataset['Month'] == 9) & (dataset['Day'] == 14),(dataset['Month'] == 9) & (dataset['Day'] == 15),
    (dataset['Month'] == 9) & (dataset['Day'] == 16),(dataset['Month'] == 9) & (dataset['Day'] == 17),(dataset['Month'] == 9) & (dataset['Day'] == 18),
    (dataset['Month'] == 9) & (dataset['Day'] == 19)
    ]
# create a list of the values we want to assign for each condition
valueweather = ['38','38','39', '39', '42','40', '38', '38', '38' ,'38',
                '37','38','39','40', '42','43', '41','41', '42','42','42', '43', '43', '42',
                '42', '42', '41', '40', '39', '40', '42', '42','42', '41','40'
                ]
# create a new column and use np.select to assign values to it using our lists as arguments
dataset['high temperature in °C'] = np.select(conditionweather, valueweather)
# Display a random sampling of 5 rows
print(dataset.sample(5, random_state=0))

#first figure
plt.figure(1)
sns.kdeplot(dataset.groupby('Location')['Volume (m³)'].mean(),color='green',shade=True)
plt.xlabel('average daily water volume consumption per location [m³]')
plt.title('Density curve with summary statistics over average daily consumption')
plt.text(30, .13, r'Summary Statistics', {'color': 'black', 'fontsize': 9})
plt.text(30, .12, r'Mode = 3.49', {'color': 'black', 'fontsize': 9})
plt.text(30, .11, r'Median = 4.852', {'color': 'black', 'fontsize': 9})
plt.text(30, .10, r'Mean = 6.000', {'color': 'black', 'fontsize': 9})
plt.text(30, .09, r'Minimum = 0.251', {'color': 'black', 'fontsize': 9})
plt.text(30, .08, r'Maximum = 38.019', {'color': 'black', 'fontsize': 9})
plt.annotate('', xy=(3.49,-0.005), xytext=(30,.123), color= 'black', fontsize=9,
            arrowprops=dict(facecolor='black',arrowstyle="->")
            ,rotation=90, rotation_mode='anchor')
plt.annotate('', xy=(4.852,-0.005), xytext=(30,.113), color= 'black', fontsize=9,
            arrowprops=dict(facecolor='black',arrowstyle="->")
            ,rotation=90, rotation_mode='anchor')
plt.annotate('', xy=(6,-0.005), xytext=(30,.103), color= 'black', fontsize=9,
            arrowprops=dict(facecolor='black',arrowstyle="->")
            ,rotation=90, rotation_mode='anchor')
#added statistics to Density  plot
print(dataset['Volume (m³)'].mode())
print(dataset['Volume (m³)'].median())
print(dataset['Volume (m³)'].mean())
print(dataset['Volume (m³)'].max())
print(dataset['Volume (m³)'].min())
interactive(True)
plt.show()

#second figure histogram
plt.figure(2)
dataset.groupby('Location')['Volume (m³)'].mean().plot.hist(density=True, facecolor='g', alpha=0.75)
plt.xlabel('average daily water volume consumption [m³]')
plt.title('Histogram of average daily water volume consumption per location')
interactive(True)
plt.show()

#Bar chart visualizing seasonality water consumption statistics
print(dataset.groupby('Season')['Volume (m³)'].sum())
sum=dataset.groupby('Season')['Volume (m³)'].sum()
print(sum[0])
print(dataset.groupby('Season')['Volume (m³)'].mean())
mean=dataset.groupby('Season')['Volume (m³)'].mean()
print(dataset.groupby('Season')['Volume (m³)'].max())
max=dataset.groupby('Season')['Volume (m³)'].max()
print(dataset.groupby('Season')['Volume (m³)'].median())
median=dataset.groupby('Season')['Volume (m³)'].median()
print(dataset.groupby('Season')['Volume (m³)'].min())
min=dataset.groupby('Season')['Volume (m³)'].min()
plt.figure(3)
# set width of bar
barWidth = 0.25
bars = [sum[0], sum[1]]
bars1 = [max[0], max[1]]
bars2 = [mean[0], mean[1]]
bars3 = [median[0], median[1]]
bars4= [min[0], min[1]]

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Maximum Volume')
plt.text(-0.1, 38.2, r'38.019', {'color': 'black', 'fontsize': 9})
plt.text(0.9, 36.6, r'36.561', {'color': 'black', 'fontsize': 9})
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Mean Volume')
plt.text(0.2, 6.75, r'6.7', {'color': 'black', 'fontsize': 9})
plt.text(1.2, 5.2, r'5.1', {'color': 'black', 'fontsize': 9})
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='Median Volume')
plt.text(0.4, 5.516, r'5.516', {'color': 'black', 'fontsize': 9})
plt.text(1.4, 4.184, r'4.184', {'color': 'black', 'fontsize': 9})
plt.bar(r4, bars4, color='Red', width=barWidth, edgecolor='white', label='Minimum Volume')
plt.text(0.65, 0.325, r'0.321', {'color': 'black', 'fontsize': 9})
plt.text(1.7, 0.255, r'0.251', {'color': 'black', 'fontsize': 9})
# Add xticks on the middle of the group bars
plt.xlabel('Seasons')
plt.ylabel('Water volume consumption [m³]')
plt.title('Bar chart visualizing seasonality water consumption statistics')
plt.xticks([r + barWidth for r in range(len(bars1))], ['Fall', 'Summer'])
plt.legend()
interactive(True)
plt.show()

#The total water volume consumption Vs seasons
plt.figure(4)
plt.bar(['Fall', 'Summer'], bars, align='center')
plt.text(0.8, 7.114606e+06, r'7.114606e+06', {'color': 'black', 'fontsize': 9})
plt.text(-0.15, 1.044275e+07, r'1.044275e+07', {'color': 'black', 'fontsize': 9})
plt.xlabel('Seasons')
plt.ylabel('Water volume consumption [m³]')
plt.title('The total water volume consumption Vs seasons')
interactive(True)
plt.show()

#The daily variations over the weekdays of average consumption
weekday_mean=dataset.groupby('Weekday Name')['Volume (m³)'].mean()
print(weekday_mean)
weekday_bars = [weekday_mean[3], weekday_mean[1], weekday_mean[5], weekday_mean[6], weekday_mean[4], weekday_mean[0], weekday_mean[2]]
weekday_r1 = np.arange(len(weekday_bars))
plt.figure(5)
plt.bar(weekday_r1, weekday_bars, align='center')
plt.xticks(weekday_r1 , ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], fontsize=8)
plt.xlabel('Weekdays')
plt.ylabel('Water volume consumption [m³]')
plt.title('The daily variations over the weekdays of average consumption')
interactive(True)
plt.show()

Hour_mean=dataset.groupby('Hour')['Volume (m³)'].mean()
print(Hour_mean)
Hour_bars = [Hour_mean[0], Hour_mean[1], Hour_mean[2], Hour_mean[3], Hour_mean[4], Hour_mean[5], Hour_mean[6],
                Hour_mean[7], Hour_mean[8], Hour_mean[9], Hour_mean[10], Hour_mean[11], Hour_mean[12], Hour_mean[13],
                Hour_mean[14], Hour_mean[15], Hour_mean[16], Hour_mean[17], Hour_mean[18], Hour_mean[19], Hour_mean[20],
                Hour_mean[21], Hour_mean[22], Hour_mean[23]]
Hour_r1 = np.arange(len(Hour_bars))
#Hourly variations of average consumption visualized as a bar chart
plt.figure(6)
plt.bar(Hour_r1, Hour_mean , align='center')
plt.xticks(Hour_r1, ['0', '1', '2', '3', '4', '5', '6',
                      '7', '8', '9', '10', '11', '12', '13',
                      '14', '15', '16', '17', '18', '19', '20',
                      '21', '22', '23'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('Hourly variations of average consumption visualized as a bar chart')
interactive(True)
plt.show()

# 1 location water consumption
plt.figure(7)
dHour=dataset.groupby(['Location', 'Hour'])['Volume (m³)'].mean()
plt.plot(Hour_r1,dHour[:24])
plt.xticks(Hour_r1 , ['0', '1', '2', '3', '4', '5', '6',
                      '7', '8', '9', '10', '11', '12', '13',
                      '14', '15', '16', '17', '18', '19', '20',
                      '21', '22', '23'], fontsize=8)
interactive(True)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('Hourly variations of average water consumption for one location')
plt.show()


# 10 location water consumption
print(dataset.iloc[[149112] , :])
plt.figure(8)
dataset10=dataset[:149111]
Hour10_mean=dataset10.groupby('Hour')['Volume (m³)'].mean()
print(Hour10_mean)
plt.plot(Hour_r1,Hour10_mean)
plt.xticks(Hour_r1 , ['0', '1', '2', '3', '4', '5', '6',
                      '7', '8', '9', '10', '11', '12', '13',
                      '14', '15', '16', '17', '18', '19', '20',
                      '21', '22', '23'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('Hourly variations of average water consumption for 10 locations')
interactive(True)
plt.show()

# water consumption for all the locations
plt.figure(9)
plt.plot(Hour_r1, Hour_mean)
plt.xticks(Hour_r1 , ['0', '1', '2', '3', '4', '5', '6',
                      '7', '8', '9', '10', '11', '12', '13',
                      '14', '15', '16', '17', '18', '19', '20',
                      '21', '22', '23'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('Hourly variations of average water consumption for all the locations')
interactive(True)
plt.show()

#The daily variations over the weekdays of average consumption
plt.figure(10)
plt.plot(weekday_r1, weekday_bars)
plt.xticks(weekday_r1 , ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], fontsize=8)
plt.xlabel('Weekdays')
plt.ylabel('Water volume consumption [m³]')
plt.title('The daily variations over the weekdays of average consumption')
interactive(True)
plt.show()


#print the plots for Quarters water consumption
Quarter_mean=dataset.groupby('Quarter')['Volume (m³)'].mean()
print(Quarter_mean)
Quarter_r1 = np.arange(2)

# 1 location water consumption
plt.figure(11)
dQuarter=dataset.groupby(['Location', 'Quarter'])['Volume (m³)'].mean()
#plt.plot(Quarter_r1,dQuarter[:2])
plt.bar(Quarter_r1, dQuarter[:2] , align='center')
plt.xticks(Quarter_r1 , ['00 to 11:59 AM', '12 to 11:59 PM'], fontsize=8)
interactive(True)
plt.xlabel('Quarters')
plt.ylabel('Water volume consumption [m³]')
plt.title('Quarter variations of average water consumption for one location')
plt.show()


# 10 location water consumption
plt.figure(12)
Quarter10_mean=dataset10.groupby('Quarter')['Volume (m³)'].mean()
print(Quarter10_mean)
#plt.plot(Quarter_r1,Quarter10_mean)
plt.bar(Quarter_r1, Quarter10_mean , align='center')
plt.xticks(Quarter_r1 , ['00 to 11:59 AM', '12 to 11:59 PM'], fontsize=8)
plt.xlabel('Quarters')
plt.ylabel('Water volume consumption [m³]')
plt.title('Quarter variations of average water consumption for 10 locations')
interactive(True)
plt.show()

# water consumption for all the locations
plt.figure(13)
#plt.plot(Quarter_r1, Quarter_mean)
plt.bar(Quarter_r1, Quarter_mean , align='center')
plt.xticks(Quarter_r1, ['00 to 11:59 AM', '12 to 11:59 PM'], fontsize=8)
plt.xlabel('Quarters')
plt.ylabel('Water volume consumption [m³]')
plt.title('Quarter variations of average water consumption for all the locations')
interactive(True)
plt.show()

#plot of water consumption for one location
plt.figure(14)
plt.plot(Quarter_r1,dQuarter[:2])
plt.xticks(Quarter_r1 , ['00 to 11:59 AM', '12 to 11:59 PM'], fontsize=8)
interactive(True)
plt.xlabel('Quarters')
plt.ylabel('Water volume consumption [m³]')
plt.title('Quarter variations of average water consumption for one location')
plt.show()

#plot of 10 location water consumption
plt.figure(15)
plt.plot(Quarter_r1,Quarter10_mean)
plt.xticks(Quarter_r1 , ['00 to 11:59 AM', '12 to 11:59 PM'], fontsize=8)
plt.xlabel('Quarters')
plt.ylabel('Water volume consumption [m³]')
plt.title('Quarter variations of average water consumption for 10 locations')
interactive(True)
plt.show()

#plot of water consumption for all the locations
plt.figure(16)
plt.plot(Quarter_r1, Quarter_mean)
plt.xticks(Quarter_r1, ['00 to 11:59 AM', '12 to 11:59 PM'], fontsize=8)
plt.xlabel('Quarters')
plt.ylabel('Water volume consumption [m³]')
plt.title('Quarter variations of average water consumption for all the locations')
interactive(True)
plt.show()

#regression analysis
plt.figure(17)
Sum_con=dataset.groupby(['Month', 'Day'])['Volume (m³)'].mean()
Sum_cons=Sum_con.to_numpy()
Sum_cons_values = [Sum_cons[10],Sum_cons[0],Sum_cons[1],Sum_cons[6],Sum_cons[7],Sum_cons[8],Sum_cons[9],Sum_cons[11],
                   Sum_cons[2],Sum_cons[3],Sum_cons[12],Sum_cons[28],Sum_cons[5],Sum_cons[13],Sum_cons[27],Sum_cons[29],
                   Sum_cons[34],Sum_cons[16],Sum_cons[17],Sum_cons[26],Sum_cons[33],Sum_cons[4],Sum_cons[14],Sum_cons[18],
                   Sum_cons[19],Sum_cons[20],Sum_cons[23],Sum_cons[24],Sum_cons[25],Sum_cons[30],Sum_cons[31],Sum_cons[32],
                   Sum_cons[15],Sum_cons[21],Sum_cons[22]
                   ]

yi_values = [valueweather[10],valueweather[0],valueweather[1],valueweather[6],valueweather[7],valueweather[8],valueweather[9],valueweather[11],
                   valueweather[2],valueweather[3],valueweather[12],valueweather[28],valueweather[5],valueweather[13],valueweather[27],valueweather[29],
                   valueweather[34],valueweather[16],valueweather[17],valueweather[26],valueweather[33],valueweather[4],valueweather[14],valueweather[18],
                   valueweather[19],valueweather[20],valueweather[23],valueweather[24],valueweather[25],valueweather[30],valueweather[31],valueweather[32],
                   valueweather[15],valueweather[21],valueweather[22] ]
yii_values= np.arange(len(yi_values))
X = np.array(list(map(int,yi_values)))
plt.scatter(X, Sum_cons_values, color='green')
plt.xlabel('The daily weather high temperature in °C')
plt.ylabel('Water volume consumption [m³]')
plt.title('The Weather Vs The Water volume consumption')
interactive(True)
plt.show()

#regression analysis with line and statistics
plt.figure(18)
Y = np.array(Sum_cons_values)
#Regression Analysis
model = LinearRegression()
model.fit(X.reshape((-1, 1)), Y.reshape((-1, 1)))
model = LinearRegression().fit(X.reshape((-1, 1)),Y.reshape((-1, 1)),)
r_sq = model.score(X.reshape((-1, 1)), Y.reshape((-1, 1)))
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)
y_pred = model.predict(X.reshape((-1, 1)))
plt.scatter(X, Sum_cons_values, color='green')
plt.xlabel('The daily weather high temperature in °C')
plt.ylabel('Water volume consumption [m³]')
plt.title('The Weather Vs The Water volume consumption')
plt.plot( X , y_pred )
plt.text(37, 7.5, r'Regression Statistics', {'color': 'black', 'fontsize': 9})
plt.text(37, 7.3, r'R Square = 0.311', {'color': 'black', 'fontsize': 9})
plt.text(37, 7.1, r'Intercept = -6.231', {'color': 'black', 'fontsize': 9})
plt.text(37, 6.9, r'Slope = 0.302', {'color': 'black', 'fontsize': 9})
interactive(False)
plt.show()

