#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pycaret.datasets import get_data


# In[2]:


dataset = get_data('credit')


# In[3]:


dataset.shape


# In[4]:


data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[5]:


from pycaret.classification import *


# In[6]:


exp_clf101 = setup(data = data, target = 'default', session_id=123)


# # Comparing All Models

# In[7]:


best_model = compare_models()


# In[8]:


print(best_model)


# # Create a Model

# In[9]:


models()


# # Decision Tree Classifier

# In[10]:


dt = create_model('dt')


# In[11]:


print(dt)


# #  K Neighbors Classifier

# In[12]:


knn = create_model('knn')


# # Random Forest Classifier

# In[13]:


rf = create_model('rf')


# # Tune a Model Decision Tree Classifier

# In[14]:


tuned_dt = tune_model(dt)


# # Tune a Model K Neighbors Classifier

# In[15]:


import numpy as np


# In[16]:


tuned_knn = tune_model(knn, custom_grid = {'n_neighbors' : np.arange(0,50,1)})


# In[17]:


print(tuned_knn)


# # Tune a Model Random Forest Classifier

# In[19]:


tuned_rf = tune_model(rf)


# #  AUC Plot

# In[20]:


plot_model(tuned_rf, plot = 'auc')


# # Precision-Recall Curve

# In[21]:


plot_model(tuned_rf, plot = 'pr')


# #  Feature Importance Plot

# In[22]:


plot_model(tuned_rf, plot='feature')


# #  Confusion Matrix

# In[23]:


plot_model(tuned_rf, plot = 'confusion_matrix')


# In[24]:


evaluate_model(tuned_rf)


# In[25]:


predict_model(tuned_rf);


# # Finalize Model for Deployment

# In[29]:


final_rf = finalize_model(tuned_rf)


# In[30]:


print(final_rf)


# In[31]:


predict_model(final_rf);


# #  Predict on unseen data

# In[32]:


unseen_predictions = predict_model(final_rf, data=data_unseen)


# In[33]:


unseen_predictions.head()


# In[34]:


from pycaret.utils import check_metric


# In[35]:


check_metric(unseen_predictions['default'], unseen_predictions['Label'], metric = 'Accuracy')


# # Saving the model

# In[36]:


save_model(final_rf,'Final RF Model 23Dec2020')


# #  Loading the saved model

# In[38]:


saved_final_rf = load_model('Final RF Model 23Dec2020')


# In[39]:


new_prediction = predict_model(saved_final_rf, data=data_unseen)


# In[40]:


new_prediction.head()


# In[41]:


from pycaret.utils import check_metric


# In[42]:


check_metric(new_prediction['default'], new_prediction['Label'], metric = 'Accuracy')


# In[ ]:




