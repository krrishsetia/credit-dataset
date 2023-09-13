import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing

pd.options.display.max_columns = 5
pd.options.display.max_rows = 10000000

data = pd.read_csv('csv files/credit_data.csv')

def gender(var):
    if var == 'Male':
        return 0
    elif var == 'Female':
        return 1
    elif var == 'Other':
        return 2

def customer(var):
    if var == 'Yes':
        return 0
    elif var == 'No':
        return 1

def state(var):
    if var == 'Delhi':
        return 0
    elif var == 'Gujarat':
        return 1
    elif var == 'Karnataka':
        return 2
    elif var == 'Kerala':
        return 3
    elif var == 'Maharashtra':
        return 4
    elif var == 'Rajasthan':
        return 5
    elif var == 'Tamil Nadu':
        return 6
    elif var == 'Telangana':
        return 7
    elif var == 'Uttar Pradesh':
        return 8
    elif var =='West Bengal':
        return 9

def pay(var):
    if var == 'Student':
        return 0
    elif var == 'Self-Employed':
        return 1
    elif var == 'Salaried':
        return 2
    elif var == 'Freelancer':
        return 3
    elif var == 'Unemployed':
        return 4

def job(var):
    if var == 'Banker':
        return 0
    elif var == 'Business Owner':
        return 1
    elif var == 'Civil Servant':
        return 2
    elif var == 'Contractor':
        return 3
    elif var == 'Doctor':
        return 4
    elif var == 'Farmer':
        return 5
    elif var == 'Graphic Designer':
        return 6
    elif var == 'Independent Consultant':
        return 7
    elif var == 'Photographer':
        return 8
    elif var == 'Shopkeeper':
        return 9
    elif var == 'Software Engineer':
        return 10
    elif var == 'Student':
        return 11
    elif var == 'Teacher':
        return 12
    elif var == 'Writer':
        return 13
    elif var =='NA':
        return 14
    else:return 14
data['Employment Profile'] = data['Employment Profile'].apply(pay)
data['State'] = data['State'].apply(state)
data['Gender'] = data['Gender'].apply(gender)
data['Existing Customer'] = data['Existing Customer'].apply(customer)
data['Occupation'] = data['Occupation'].apply(job)
data.drop(columns='City',axis=1,inplace=True)

x = data['Age'].values.reshape(-1,1)
y = data['State'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=125)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train,y_train)
x_test_scaled = scaler.transform(x_test)

regression = KNeighborsClassifier()
regression.fit(x_train_scaled,y_train)
y_pred = regression.predict(x_test_scaled)

plt.scatter(x_test_scaled,y_test)
plt.plot(x_test_scaled,y_pred)
plt.show()
y_pred = np.round(y_pred)
print('Mean squared error:',metrics.mean_squared_error(y_true=y_test,y_pred=y_pred))
print('Regular accuracy:',metrics.accuracy_score(y_true=y_test,y_pred=y_pred)*100,'%')
print('Balanced accuracy:',metrics.balanced_accuracy_score(y_true=y_test,y_pred=y_pred)*100,'%')
print('F1:',metrics.f1_score(y_true=y_test,y_pred=y_pred,average='macro')*100,'%')
print('Precision:',metrics.precision_score(y_true=y_test,y_pred=y_pred,average='macro')*100,'%')
print('Kappa:',metrics.cohen_kappa_score(y1=y_test,y2=y_pred)*100,'%')

matrix = metrics.confusion_matrix(y_true=y_test,y_pred=y_pred)
display = metrics.ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()