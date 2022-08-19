#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
sns.set(style="whitegrid")
np.random.seed(203)


# In[2]:


#Preprocesing
data = pd.read_csv("framingham.csv")


# In[7]:


data.head()


# In[6]:


import numpy as np
data = data.replace('?', np.nan)

data.isnull().sum()

data = data.fillna(axis=0, method='ffill')
data.isnull().sum()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go


# In[9]:


data.dtypes


# In[11]:


vc = data['TenYearCHD'].value_counts().to_frame().reset_index()


# In[9]:


vc


# In[8]:


vc['percent'] = vc["TenYearCHD"].apply(lambda x : round(100*float(x) / len(data), 2))
vc = vc.rename(columns = {"index" : "Target", "Class" : "Count"})
vc


# In[9]:


non_CHD = data[data['TenYearCHD'] == 0]
CHD = data[data['TenYearCHD'] == 1]


# In[10]:


df = non_CHD.append(CHD).reset_index(drop=True)
X = df.drop(['TenYearCHD'], axis = 1).values
Y = df["TenYearCHD"].values
print(X.dtype)
print(Y.dtype)


# In[27]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras import backend as K


class MyRegularizer(regularizers.Regularizer):

    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):
        return self.strength * tf.math.reduce_sum(tf.math.square(x) * tf.math.abs(x)) 

    def get_config(self):
        return {'strength': self.strength}


# In[28]:


#Autoencoder Model

## input layer 
input_layer = Input(shape=(X.shape[1],))

## encoding part
encoded = Dense(100, activation='tanh', activity_regularizer=MyRegularizer(0.02))(input_layer)
encoded = Dense(50, activation='relu')(encoded)

## decoding part
decoded = Dense(50, activation='tanh')(encoded)
decoded = Dense(100, activation='tanh')(decoded)

## output layer
output_layer = Dense(X.shape[1], activation='relu')(decoded)


# In[29]:


autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")


# In[30]:


x = data.drop(["TenYearCHD"], axis=1)
y = data["TenYearCHD"].values

x_scale = preprocessing.MinMaxScaler().fit_transform(x.values)

x_norm, x_CHD = x_scale[y == 0], x_scale[y == 1]


# In[31]:


x_norm.shape


# In[32]:


autoencoder.fit(x_norm, x_norm, 
                batch_size = 40, epochs = 50, 
                shuffle = True, validation_split = 0.20);


# In[16]:


hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])


# In[17]:


norm_hid_rep = hidden_representation.predict(x_norm)
CHD_hid_rep = hidden_representation.predict(x_CHD)


# In[18]:


rep_x = np.append(norm_hid_rep, CHD_hid_rep, axis = 0)

y_n = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(CHD_hid_rep.shape[0])

rep_y = np.append(y_n, y_f)

X_train, X_test, y_train, y_test = train_test_split(rep_x, rep_y, test_size=0.25, random_state=0)


# In[19]:


print(rep_x.shape)


# In[20]:


rep_x = pd.DataFrame(rep_x)
rep_x


# In[21]:


rep_x.to_csv("Autoencoded_feature_data.csv", index=False, header=True)


# In[22]:


#classification model

# Initialising the model
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(64, activation = 'relu', input_dim = 50))

# Adding the second hidden layer - apply grid search to find the optimal number of layers 
classifier.add(Dense(32, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(1, activation = 'sigmoid')) 

# Compiling the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

# Fitting the model to the Training set
classifier.fit(X_train, y_train, batch_size = 20, epochs = 100) 


# In[23]:


classifier.evaluate(X_train, y_train)


# In[24]:


classifier.evaluate(X_test, y_test)


# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import selectFromModel
from sklearn.model_selection import RandomForestClassifier


# In[ ]:


from sklearn.model_selection import RFE
import hybrid
sel_= RFE()


# In[ ]:




