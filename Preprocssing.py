# Importred Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dask import dataframe as dd
import re

# Set themes for my graphs
sns.set_style('darkgrid')
sns.set_theme(style="darkgrid")

# Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_rows', None)


# Display all rows and columns of a dataframe instead of a truncated version
from IPython.display import display
pd.set_option('display.max_rows', None)

# check how much we use from memory
def memory_usage(data):
    """ Check the momery. 
    """
    print('Memory usage')
    print(data.info(memory_usage=True)) 

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
    
    train = dd.read_csv(df1, dtype = types_data)
    test = pd.read_csv(df2, dtype = types_data2)
    
    return train,test
data, test = load_data('train.csv','test.csv')

# Import our data 
client = pd.read_csv('client.csv')
town = pd.read_csv('town.csv')
product = pd.read_csv('product.csv')

# change the name of columns 
data.columns = ['Week Number', 'Sales Depot Id', 'Sales Channel ID', 'Route ID', 'Client ID', 'Product ID',
              'Sales Unit This Week','Sales This Week', 'Returns Unit Next Week', 'Returns Next Week',
              'Adjusted Demand']

# change the name of columns 
test.drop('id', axis=1, inplace=True)
test.columns = ['Week Number', 'Sales Depot Id', 'Sales Channel ID', 'Route ID', 'Client ID', 'Product ID']


# deal with data and convert it to datframe 
number_week = data['Week Number'].nunique().compute()


# preproccseing 
# I will Work on the Week 3 the first one to check the data
df_week3 = data.head(11165207)

# understanding our data by small steps and visual tools
def understand_our_data(data):
    print('Shape of our Data','\n')
    print(data.shape,'\n')
    print('Our Data Info','\n')
    print(data.info(),'\n')
    print('Describe our Numeric data','\n')
    print(data.describe(),'\n')
    print('Number of unique vales','\n')
    print(data.nunique().sort_values(),'\n')
    print('percantge of null values', '\n')
    print(round(data.isna().sum(axis=0)/len(data),2)*100)
    
understand_our_data(df_week3)

# Plot Our data
def scatter_plot(df, columnx, columny):
    plt.scatter(df[columnx], df[columny])
    plt.title(columnx)
    plt.show()

for col in df_week3.columns:
    scatter_plot(df_week3,col,'Adjusted Demand')
    
# Check Week 9 the last one
df_week9 = data.tail(10191837)

understand_our_data(df_week9)
for col in df_week9.columns:
    scatter_plot(df_week9,col,'Adjusted Demand')