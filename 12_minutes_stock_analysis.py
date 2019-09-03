import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

start = datetime.datetime(2019,8,1)
end = datetime.datetime(2019,8,12)

df = web.DataReader("AAPL", 'yahoo', start, end)
df.tail()

#Let's start code out the Rolling Mean
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

import matplotlib.pyplot as plt
from matplotlib import style

#Adjusting the size of matplotlib
import matplotlib as mpl
mpl.rc('figure', figsize=(8,7))
mpl.__version__

#Adjusting the style of matplotlib
style.use('ggplot')

close_px.plot(label='AAPL')

mavg.plot(label='mavg')
plt.legend()

rets = close_px / close_px.shift(1) - 1

rets.plot(label='return')

dfcomp = web.DataReader(['AAPL','GE','GOOG','IBM','MSFT'],'yahoo',start=start,end=end)['Adj Close']

retscomp = dfcomp.pct_change()

corr = retscomp.corr()

#Let's plot Apple and GE with ScatterPlot to view their return distributions.
plt.scatter(retscomp.AAPL, retscomp.GE)
plt.xlabel('Returns AAPL')
plt.ylabel('Returns GE')

pd.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));

plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);

#Stocks Returns rate and Risk
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
            label,
            xy = (x, y), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom', bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
#Predicting stocks price
dfreg = df.loc[:,['Adj Close', 'volume']]
dfreg['HL_PCT'] = (df['High']-df['Low']) / df['Close'] * 100.0
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

#Drop missing values
import math
import numpy as np
from sklearn import preprocessing

dfreg.fillna(value=-99999, inplace=True)

#we want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

#Separating the label here, we want to predict the AdjClose
forecast_col='Adj Close'
dfreg['label']=dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'], 1))

#Scale the X so that everyone can have the same distribution for linear regression
X = preprocessing.scale(X)

#Finally we want to Data Series of late X and early X (train) model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#Separate label and identify it as Y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

#Model Generation-where the prediction fun starts
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Spliting the dataset inti training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fitting the Classifier to the Training Set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
classifier = MLPRegressor()
classifier = KNeighborsClassifier(n_neighbors =3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

#K nearest KNN Regression
from sklearn.neighbors import KNeighborsClassifier
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit = (X_train, y_train)

#Evaluation
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

#For sanity testing, let us print some of the stocks forecast.
forecast_set = clf.predict(X_lately)
dfreg['Forecast'] = np.nan

#Plotting the Prediction
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

