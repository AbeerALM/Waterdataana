import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib import interactive
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans

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


#rearange the features column based on the time column
ns15min=15*60*1000000000   # 15 minutes in nanoseconds
dataset['Features'] = pd.to_datetime(((dataset.index.astype(np.int64) // ns15min + 1 ) * ns15min))
dataset['DateToCluster'] = dataset.index.date
feature=dataset.reset_index()
temp = pd.DatetimeIndex(feature['Features'])
feature['FeatureTime'] = temp.time
del feature['Features']
feature=feature.groupby(['DateToCluster', 'Location','FeatureTime'])['Volume (m³)'].mean().reset_index()
feature['Date-Location'] = feature['DateToCluster'].astype(str) +' '+ feature['Location']
del feature['DateToCluster']
del feature['Location']
fg=pd.pivot_table(feature, values='Volume (m³)', index='Date-Location', columns='FeatureTime',fill_value=0)

fgY = np.array(fg)
formatted_dataset = to_time_series_dataset(fgY)



seed=0
dba_km = TimeSeriesKMeans(n_clusters=3,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          random_state=seed,
                          max_iter_barycenter=10)
#plot one day clusters
plt.figure(1)
X_train=formatted_dataset[0:199,:]
y_pred = dba_km.fit_predict(X_train)
sz = X_train.shape[1]
for yi in range(3):
    plt.subplot(3, 3, 4 + yi)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.xticks([0,25,50,75], ['0', '5:45','12', '18:15'], fontsize=8)
    #plt.ylim(-2, 22)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)

    if yi == 1:
        plt.title("DBA $k$-means hour variations of average water consumption for one day/different locations")
interactive(True)
plt.show()

#plot for first segment
plt.figure(2)
X_train=formatted_dataset[0:1717,:]
y_pred = dba_km.fit_predict(X_train)
sz = X_train.shape[1]
plt.plot(dba_km.cluster_centers_[0].ravel(), "r-")
plt.plot(dba_km.cluster_centers_[1].ravel(), "b-")
plt.plot(dba_km.cluster_centers_[2].ravel(), "y-")
plt.xlim(0, sz)
plt.xticks([0,25,50,75], ['0', '5:45','12', '18:15'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('DBA $k$-means hour variations of average water consumption for first segment')
interactive(True)
plt.show()
#plot for second segment
plt.figure(3)
X_train=formatted_dataset[1718:3435,:]
y_pred = dba_km.fit_predict(X_train)
sz = X_train.shape[1]
plt.plot(dba_km.cluster_centers_[0].ravel(), "r-")
plt.plot(dba_km.cluster_centers_[1].ravel(), "b-")
plt.plot(dba_km.cluster_centers_[2].ravel(), "y-")
plt.xlim(0, sz)
plt.xticks([0,25,50,75], ['0', '5:45','12', '18:15'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('DBA $k$-means hour variations of average water consumption for second segment')
interactive(True)
plt.show()
#plot for third segment
plt.figure(4)
X_train=formatted_dataset[3436:5153,:]
y_pred = dba_km.fit_predict(X_train)
sz = X_train.shape[1]
plt.plot(dba_km.cluster_centers_[0].ravel(), "r-")
plt.plot(dba_km.cluster_centers_[1].ravel(), "b-")
plt.plot(dba_km.cluster_centers_[2].ravel(), "y-")
plt.xlim(0, sz)
plt.xticks([0,25,50,75], ['0', '5:45','12', '18:15'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('DBA $k$-means hour variations of average water consumption for third segment')
interactive(True)
plt.show()
#plot for fforth segment
plt.figure(5)
X_train=formatted_dataset[5154:6871,:]
y_pred = dba_km.fit_predict(X_train)
sz = X_train.shape[1]
plt.plot(dba_km.cluster_centers_[0].ravel(), "r-")
plt.plot(dba_km.cluster_centers_[1].ravel(), "b-")
plt.plot(dba_km.cluster_centers_[2].ravel(), "y-")
plt.xlim(0, sz)
plt.xticks([0,25,50,75], ['0', '5:45','12', '18:15'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('DBA $k$-means hour variations of average water consumption for forth segment')
interactive(True)
plt.show()

#plot of clustring all data
plt.figure(6)
X_train=formatted_dataset
y_pred = dba_km.fit_predict(X_train)
sz = X_train.shape[1]
plt.plot(dba_km.cluster_centers_[0].ravel(), "r-")
plt.plot(dba_km.cluster_centers_[1].ravel(), "b-")
plt.plot(dba_km.cluster_centers_[2].ravel(), "y-")
plt.xlim(0, sz)
plt.xticks([0,25,50,75], ['0', '5:45','12', '18:15'], fontsize=8)
plt.xlabel('Hours')
plt.ylabel('Water volume consumption [m³]')
plt.title('DBA $k$-means hour variations of average water consumption')
interactive(False)
plt.show()