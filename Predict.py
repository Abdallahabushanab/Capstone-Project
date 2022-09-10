# Importred Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dask import dataframe as dd
import re
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error

# Set themes for my graphs
sns.set_style('darkgrid')
sns.set_theme(style="darkgrid")

# Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_rows', None)


# Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_rows', None)



# function to import our data
def load_data(df1, df2):
    # better types of data for memory 
    types_data = {
                    'Semana': np.uint8, 'Agencia_ID': np.uint16, 'Canal_ID': np.uint8
                    ,'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32, 'Producto_ID': np.uint16
                    ,'Demanda_uni_equil': np.uint16
                    }  

    types_data2 = {
                    'Semana': np.uint8, 'Agencia_ID': np.uint16, 'Canal_ID': np.uint8
                    ,'Ruta_SAK': np.uint16, 'Cliente_ID': np.uint32, 'Producto_ID': np.uint16
                    }  
    
    train = pd.read_csv(df1, dtype = types_data)
    test = pd.read_csv(df2, dtype = types_data2)
    
    return train,test

# check how much we use from memory
def memory_usage(data):
    """ Check the momery. 
    """
    print('Memory usage')
    print(data.info(memory_usage=True))   
    

    
# Import our data
train, test = load_data('Train_edited_data.csv', 'Test_edited_data.csv')



X = train.drop('Adjusted Demand', axis=1)
y= train['Adjusted Demand']
sns.heatmap(train.corr())
scaler = StandardScaler()
X = scaler.fit_transform(X) 

X_22 = test.drop('Adjusted Demand', axis=1)
y_22= test['Adjusted Demand']

del train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) 


xgb_params = {
             'objective': 'reg:linear'
             , 'n_estimators': 100 
             , 'learning_rate': 0.25 
             , 'max_depth': 12 
             , 'seed': 0
             }
model_xgb = XGBRegressor()                
model_xgb.set_params(**xgb_params)
model_xgb.fit(X_train,y_train)
y_pred = model_xgb.predict(X_test)
r2_score(y_test, y_pred)
joblib.dump(model_xgb, 'xgboost')
mean_squared_error(y_test, y_pred.astype(int))








