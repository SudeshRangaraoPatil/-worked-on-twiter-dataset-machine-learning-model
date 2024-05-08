#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[11]:


df = pd.read_csv('Twitter_Data.csv')[:5000]


# In[12]:


df.sample(10)


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.isnull().sum()


# In[17]:


df.dropna(inplace=True)


# In[31]:


df.groupby('category').count()


# In[32]:


vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
y = df['category']


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# In[34]:


knn = KNeighborsClassifier(n_neighbors=100)


# In[35]:


knn.fit(X_train, y_train)


# In[36]:


y_pred = knn.predict(X_test)


# In[37]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[38]:


new_text = ["I love this movie! It's amazing."]
new_text_vectorized = vectorizer.transform(new_text)
predicted_sentiment = knn.predict(new_text_vectorized)
print("Predicted Sentiment:", predicted_sentiment)


# In[39]:


new_text = ["I hate you dint call me"]
new_text_vectorized = vectorizer.transform(new_text)
predicted_sentiment = knn.predict(new_text_vectorized)
print("Predicted Sentiment:", predicted_sentiment)

