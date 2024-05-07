#!/usr/bin/env python
# coding: utf-8
# The main objective of these companies toward their customers is to deliver the food at the right time. To deliver the food faster, these companies identify areas where the demand for online food orders is high and employ more delivery partners in those locations. It helps deliver food faster in areas with more orders.
# In[1]:


# importing libreries


# In[2]:


import numpy as np
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("zomato.csv")
print(data.head())


# So the dataset contains information like:
# 
# the age of the customer
# marital status of the customer
# occupation of the customer
# monthly income of the customer
# educational qualification of the customer
# family size of the customer
# latitude and longitude of the location of the customer
# pin code of the residence of the customer
# did the customer order again (Output)
# Feedback of the last order (Positive or Negative)

# In[3]:


print(data.info())


# In[4]:


data.head(5)


# In[5]:


data.isna().sum()


# In[6]:


data.drop('Unnamed: 12', axis=1, inplace=True)


# In[7]:


data.drop(['longitude','latitude'],axis=1,inplace=True)


# In[8]:


#Now let’s move to the analysis of this data. I will start by looking at the online food order decisions based on the age of the customer:


# In[9]:


plt.figure(figsize=(15,10))
plt.title("online food order decision based on the age")
sns.histplot(x='Age', data=data)
plt.show()


# We can see that the age group of 22-25 ordered the food often again. It also means this age group is the target of online food delivery companies. Now let’s have a look at the online food order decisions based on the size of the family of the customer:

# In[10]:


plt.figure(figsize=(15,10))
plt.title("online food delivery based on family size")
sns.histplot(x='Family size',data=data)
plt.show()


# In[11]:


#Families with 2 and 3 members are ordering food often. These can be roommates, couples, or a family of three.


# In[12]:


data


# In[13]:


#Let’s create a dataset of all the customers who ordered the food again:


# In[14]:


buying_again_data=data.query("Output=='Yes'")
print(buying_again_data.head())


# In[15]:


print(buying_again_data)


# In[16]:


#Now let’s have a look at the gender column. Let’s find who orders food more online:


# In[17]:


gender = buying_again_data["Gender"].value_counts()
label = gender.index
counts = gender.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Who Orders Food Online More: Male Vs. Female')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


# In[18]:


#According to the dataset, male customers are ordering more compared the females. Now let’s have a look at the marital status of the customers who ordered again:


# In[19]:


marital = buying_again_data["Marital Status"].value_counts()
label = marital.index
counts = marital.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Who Orders Food Online More: Married Vs. Singles')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


# In[20]:


#According to the above figure, 76.1% of the frequent customers are singles. Now let’s have a look at what’s the income group of the customers who ordered the food again:


# In[21]:


income = buying_again_data["Monthly Income"].value_counts()
label = income.index
counts = income.values


fig = go.Figure(data=[go.Pie(labels=label, values=counts)])
fig.update_layout(title_text='Which Income Group Orders Food Online More')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


# In[22]:


#According to the above figure, 54% of the customers don’t fall under any income group. They can be housewives or students.


# In[23]:


data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["Marital Status"] = data["Marital Status"].map({"Married": 2, 
                                                     "Single": 1, 
                                                     "Prefer not to say": 0})
data["Occupation"] = data["Occupation"].map({"Student": 1, 
                                             "Employee": 2, 
                                             "Self Employeed": 3, 
                                             "House wife": 4})
data["Educational Qualifications"] = data["Educational Qualifications"].map({"Graduate": 1, 
                                                                             "Post Graduate": 2, 
                                                                             "Ph.D": 3, "School": 4, 
                                                                             "Uneducated": 5})
data["Monthly Income"] = data["Monthly Income"].map({"No Income": 0, 
                                                     "25001 to 50000": 5000, 
                                                     "More than 50000": 7000, 
                                                     "10001 to 25000": 25000, 
                                                     "Below Rs.10000": 10000})
data["Feedback"] = data["Feedback"].map({"Positive": 1, "Negative ": 0})
print(data.head())


# # Online Food Order Prediction Model

# Now let’s train a machine learning model to predict whether a customer will order again or not. I will start by splitting the data into training and test sets:

# In[24]:


from sklearn.model_selection import train_test_split
x = np.array(data[["Age", "Gender", "Marital Status", "Occupation", 
                   "Monthly Income", "Educational Qualifications", 
                   "Family size", "Pin code", "Feedback"]])
y = np.array(data[["Output"]])


# In[25]:


# training a machine learning model
from sklearn.ensemble import RandomForestClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)
model = RandomForestClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[26]:


#Now let’s prepare a form to input the data of the customer and predict whether the customer will order the food again or not:


# In[27]:


print("Enter Customer Details to Predict If the Customer Will Order Again")
a = int(input("Enter the Age of the Customer: "))
b = int(input("Enter the Gender of the Customer (1 = Male, 0 = Female): "))
c = int(input("Marital Status of the Customer (1 = Single, 2 = Married, 3 = Not Revealed): "))
d = int(input("Occupation of the Customer (Student = 1, Employee = 2, Self Employeed = 3, House wife = 4): "))
e = int(input("Monthly Income: "))
f = int(input("Educational Qualification (Graduate = 1, Post Graduate = 2, Ph.D = 3, School = 4, Uneducated = 5): "))
g = int(input("Family Size: "))
h = int(input("Pin Code: "))
i = int(input("Review of the Last Order (1 = Positive, 0 = Negative): "))
features = np.array([[a, b, c, d, e, f, g, h, i]])
print("Finding if the customer will order again: ", model.predict(features))


# # Summary

# So this is how you can predict whether a customer will order food online again or not. The food order prediction system is one of the useful techniques food delivery companies can use to make the entire delivery process fast.

# In[ ]:




