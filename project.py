# Importred Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dask import dataframe as dd
import re
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
stemmer = SnowballStemmer('spanish')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from dask_ml import preprocessing
import gc
import joblib

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
def load_data(df1):
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
    
    return train
data1 = load_data('train.csv')

# Import our data 
client = pd.read_csv('client.csv')
town = pd.read_csv('town.csv')
product = pd.read_csv('product.csv')

# change the name of columns 
data1.columns = ['Week Number', 'Sales Depot Id', 'Sales Channel ID', 'Route ID', 'Client ID', 'Product ID',
              'Sales Unit This Week','Sales This Week', 'Returns Unit Next Week', 'Returns Next Week',
              'Adjusted Demand']


# deal with data and convert it to datframe 
number_week = data1['Week Number'].nunique().compute()


data = data1[data1['Week Number'] <= 4]
data = data.compute()
test = data1[data1['Week Number'] == 5]
test = test.compute()

del data1

# Check client column.
# We can see that the id is almost 935K.
# put the same clients may have the diffrenet ID 311K Client only.
client.shape
client_unique = client.groupby(['NombreCliente'])['Cliente_ID'].count()

# clean client id and remove duplicate 
def dropduplicate(df):
    df_duplicate = df.duplicated(subset = ('Cliente_ID'))
    df = df[df_duplicate== False]
    return df

# Everything else bucketed into 'Individual'
def function_word(data):
    # Avoid the single-words created so far by checking for upper-case
    if (data.isupper()) and (data != "NO IDENTIFICADO"): 
        return 'Individual'
    else:
        return data

client = dropduplicate(client)

clients =  client.copy()

# Extract type of client from the name
def create_client_features(df):
    """ Takes clients data as input.  
        Creates new variable 'Client_Type' by categorizing NombreCliente. 
        Returns clients data.
    """
    
    # Remove duplicate ids
    df = dropduplicate(df)    
    
    # Create new feature
    df1 = df.copy()
    df1['Client_Type'] = df1.loc[:, 'NombreCliente']
    
    # Convert to all UPPER-CASE
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.upper()
    
    # Known Large Company / Special Group Types
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.replace('.*REMISION.*','Consignment')
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*WAL MART.*','.*SAMS CLUB.*'],'Walmart', regex=True)
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.replace('.*OXXO.*','Oxxo Store')
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.replace('.*CONASUPO.*','Govt Store')
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.replace('.*BIMBO.*','Bimbo Store')
    
    # Term search for assortment of words picked from looking at their frequencies
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*COLEG.*','.*UNIV.*','.*ESCU.*','.*INSTI.*',\
                                                        '.*PREPAR.*'],'School', regex=True)
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.replace('.*PUESTO.*','Post')
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*FARMA.*','.*HOSPITAL.*','.*CLINI.*'],'Hospital/Pharmacy', regex=True)
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*CAFE.*','.*CREMERIA.*','.*DULCERIA.*',\
                                                        '.*REST.*','.*BURGER.*','.*TACO.*', '.*TORTA.*',\
                                                        '.*TAQUER.*','.*HOT DOG.*',\
                                                        '.*COMEDOR.*', '.*ERIA.*','.*BURGU.*'],'Eatery', regex=True)
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].str.replace('.*SUPER.*','Supermarket')
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*COMERCIAL.*','.*BODEGA.*','.*DEPOSITO.*',\
                                                            '.*ABARROTES.*','.*MERCADO.*','.*CAMBIO.*',\
                                                        '.*MARKET.*','.*MART .*','.*MINI .*',\
                                                        '.*PLAZA.*','.*MISC.*','.*ELEVEN.*','.*EXP.*',\
                                                         '.*SNACK.*', '.*PAPELERIA.*', '.*CARNICERIA.*',\
                                                         '.*LOCAL.*','.*COMODIN.*','.*PROVIDENCIA.*'
                                                        ],'General Market/Mart'\
                                                       , regex=True)                                                   
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*VERDU.*','.*FRUT.*'],'Fresh Market', regex=True)
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace(['.*HOTEL.*','.*MOTEL.*'],'Hotel', regex=True)    
 
    # Filter participles
    df1.loc[:, 'Client_Type'] = df1.loc[:, 'Client_Type'].replace([
            '.*LA .*','.*EL .*','.*DE .*','.*LOS .*','.*DEL .*','.*Y .*', '.*SAN .*', '.*SANTA .*',\
            '.*AG .*','.*LAS .*','.*MI .*','.*MA .*', '.*II.*', '.*[0-9]+.*'\
                ],'Small Franchise', regex=True)
        
    return df1
        
# Works on clients file
print(' extract the type from the data')
clients = create_client_features(clients)
clients['Client_Type'] = clients['Client_Type'].map(function_word)
print("Done")
print(clients.head())


# preprocssing on product file
def create_product_features(products):
    """ Takes products data as input and builds new features. 
        Returns modified products data.
    """
    # Split NombreProducto and create new columns
    products['short_name'] = products['NombreProducto'].str.extract('^(\D*)', expand=False)
    products['brand'] = products['NombreProducto'].str.extract('^.+\s(\D+) \d+$', expand=False)
    w = products['NombreProducto'].str.extract('(\d+)(Kg|g)', expand=True)
    products['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
    products['pieces'] =  products['NombreProducto'].str.extract('(\d+)p ', expand=False).astype('float')
    products['weight_per_piece'] = products['weight'] / products['pieces']

    products['short_product_name'] = (products['short_name']
                                        .map(lambda x: " ".join([i for i in x.lower()
                                                                 .split() if i not in stopwords.words("spanish")])))    
 

    products['short_product_name'] = (products['short_product_name']
                                        .map(lambda x: " ".join([stemmer.stem(i) for i in x.lower().split()])))

    # Drop unnecessary variables
    products = products.drop(['NombreProducto', 'short_name'], axis = 1)

    return products

# Apply the preprocssing on the  file
products = create_product_features(product)
print(products.head())

# Merge client and product with both train and test data
def merge(df):
    df_new = df.merge(products, how = 'left', left_on = 'Product ID', right_on ='Producto_ID')
    df_new = df_new.merge(clients, how = 'left', left_on = 'Client ID', right_on ='Cliente_ID')
    df_new = df_new.drop(['Cliente_ID', 'Producto_ID'], axis=1)
    df_new = df_new.dropna()
    return df_new


# merge all togther on train and test
df_new = merge(data)
test_new = merge(test)
df_new.isna().sum()
test_new.isna().sum()

del data
del test
gc.collect()

# =============================================================================
# # add mean weekly frequency for id variables 
# def Count_of_id(train,test):
#     columns = ['Product ID','Client ID','Sales Depot Id', 'Sales Channel ID', 'Route ID', 'Client ID']
#     # create mean of weekly id count
#     for column in columns:
#         train_counts = pd.DataFrame({column + '_count': test[[column, 'Week Number']].groupby([column, 'Week Number']).size()}).reset_index()
#         test_counts = pd.DataFrame({column + '_count': test[[column, 'Week Number']].groupby([column, 'Week Number']).size()}).reset_index()
#         counts = train_counts.append(test_counts)
#         counts = pd.DataFrame({column + '_count': counts.groupby(column)[column + '_count'].mean()}).reset_index()
#         counts[column + '_count'] = counts[column + '_count'].astype(np.float32)
#     
#         # merge the data with train & test
#         train = train.merge(counts, how = 'left', on=column)
#         test = test.merge(counts, how='left', on = column)
#         
#     return train,test
# 
# #apply on train and test
# df_new, test_new = Count_of_id(df_new,test_new)
# print(memory_usage(df_new))
# =============================================================================

df_new = df_new.drop(['Sales Unit This Week','Sales This Week','Returns Unit Next Week', 'Returns Next Week'], axis = 1)
test_new = test_new.drop(['Sales Unit This Week','Sales This Week','Returns Unit Next Week', 'Returns Next Week'], axis = 1)

# =============================================================================
# df_new = df_new.drop(['Client_Type','short_product_name','brand','NombreCliente'], axis = 1)
# test_new = test_new.drop(['Client_Type','short_product_name','brand','NombreCliente'], axis = 1)
# =============================================================================

# df_new = df_new.dropna()
# Do the label encoder 
def encode_labels(train, test, columns): 
    """ Converts categorical features to integers in train and test data.
        Returns train and test data.
    """
      
    # Transform columns
    for column in columns: 
        
        # Perform label encoding
        le = LabelEncoder()
        le1 = LabelEncoder()
        le.fit(train[column])
        le1.fit(test[column])
        test[column] = le1.transform(test[column])
        train[column] = le.transform(train[column])
        joblib.dump(le, 'Labelincoder')
        
        
    return train, test
col = ['Client_Type','short_product_name','brand','NombreCliente']
df_new, test_new = encode_labels(df_new,test_new,col)
print(memory_usage(df_new))

print(df_new.columns)
print(df_new.isna().sum())


# export the data 

print("4. Writing to CSV...")
clients.to_csv("cliente_tabla_modified.csv", index = False, header = True)
products.to_csv("producto_tabla_modified.csv", index = False, header = True)
df_new.to_csv("Train_edited_data.csv", index = False, header = True)
test_new.to_csv("Test_edited_data.csv", index = False, header = True)

print("Complete!")




















