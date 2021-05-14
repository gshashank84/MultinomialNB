## Naive Bayes (Using NumPy and Pandas only)

- Implemented the classifier in Python using **hash tables** to store the likelihoods if the features are **continuous**.
- Supports both **Continuous** and **Discrete** features.
- Used **Laplace Smoothing** to handle the unseen cases.
- Supports datasets having **Multi-class**. 
- **Confusion Matrix** supporting multi-class classification have been added.
- **Accuracy**, **Precision** and **Recall** score metrics methods have been added.

[Click Here](https://github.com/gshashank84/NB/blob/main/NB_main.py) for the main.py file. 

## Dependencies
- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://conda.io/en/latest/miniconda.html)

## Usage
```python

from main import NB
import pandas as pd
import numpy as np

data = pd.read_csv('./dataset/mushrooms.csv')

ind = list(data.index)
np.random.shuffle(ind)

# Train:Test = 75%:25%
train_len = int(data.shape[0]*0.75)
train_ind = ind[:train_len]
training_data = data.iloc[train_ind,:]

test_ind = ind[train_len:]
testing_data = data.iloc[test_ind,:]

print('Training_data size -> {}'.format(training_data.shape))
print('Testing_data size -> {}'.format(testing_data.shape))

assert data.shape[0] ==  len(train_ind)+ len(test_ind), 'Not equal distribution'

classifier = NB(target='class',dataframe=training_data)

y_test = list(testing_data.iloc[:,0])
y_pred = classifier.predict(testing_data.iloc[:,1:])


print('Accuracy Score -> {} %'.format(round(genx.accuracy_score(y_test,y_pred),3)))
print('Precison Score -> {}'.format(round(genx.precision_score(y_test,y_pred),3)))
print('Recall Score -> {}'.format(round(genx.recall_score(y_test,y_pred),3)))

```






