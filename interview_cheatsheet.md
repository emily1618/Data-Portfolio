# Interview Cheatsheet For Data Analyst Position
##### Author: Emi Ly

##### Date: Feb 2022

##### [Tableau Dashboard] Coming Soon
#

*Overview of the concept from theory, statistics, machine learning model, SQL, and coding. I created this notebook to prepare for interview. Source is from everywhere from google search to class room lecture slides.*
### 💻 [Machine Learning](#machine-learning)
### 🛠 [Statistics](#statistics)
### 📊 [Coding](#coding)
### 📋 [SQL](#sql)
### 🧗‍♀️ [Tableau](#tableau)





## MACHINE LEARNING

**Supervised vs Unsupervised**
- Supervised: Input and output data are provided 
  - A supervised learning model produces an accurate result. It allows you to collect data or produce a data output from the previous experience. The drawback of this model is that decision boundaries might be overstrained if your training set doesn't have examples that you want to have in a class.
- Unsupervised: Input data are provided
  - In the case of images and videos, unsupervised algorithms can rapidly classify and cluster data using far fewer features than humans might specify, making data processing even faster and more efficient.
Unsupervised machine learning finds all kinds of unknown patterns in data. Also helps you to find features that can be useful for categorization. It is easier to get unlabeled data from a computer than labeled data, which needs manual intervention. Unsupervised learning solves the problem by learning the data and classifying it without any labels. 


**Missing Values**
- KNN Imputer: There are different ways to handle missing data. Some methods such as removing the entire observation if it has a missing value or replacing the missing values with mean, median, or mode values. However, these methods can waste valuable data or reduce the variability of your dataset. In contrast, KNN Imputer maintains the value and variability of your datasets, and yet it is more precise and efficient than using the average values: https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e
Needs to normalize data before applying KNN
- Fill na: dataset['Some column']= dataset['Some column'].fillna(0)
- Fill with mean: dataset['some column']=dataset['some column'].fillna((dataset['some column'].mean()))
- Fill with nearby value: dataset['some column'] = dataset[‘some volumn'].fillna(method='ffill')
- Don’t forget to df_filling_mean.fillna(value = df_filling_mean.mean(), inplace = True)


**Bias and Variance, Overfit and Underfit**
- The inability of a ML model to capture the true relationship is called “bias.” Models with high bias are unable to capture the true relationship between input and output features, and it usually leads to oversimplification of the model.
Under fit
  - An underfit model has high bias and low variance.
- A model with high variance means it fits the training data very well but does a poor job predicting the testing data. It other words, it memorizes the training data very well and is not able to predict the test data due to low generalization.
  - Over fit
  - An overfit model means it has high variance and low bias.
  - ![fit](https://user-images.githubusercontent.com/62857660/155050192-fa6ff06c-5271-43a9-8054-cdb5464b0404.jpg)


**Dimension Reduction**
- Dimensionality reduction is the process of reducing the number of variables by obtaining a set of important variables.
- PCA 

**Flow**
1. EDA on data
2. Detect outliers
3. Extract features
  - Use domain expertise
Feature ranking, selection
Feature collinearity
If the features are collinear, providing the model with the same information could potentially result in model confusion. Simply drop one of the collinear inputs. If both inputs are important to understand, it is advised to train two separate models with each collinear feature
Removing zero-variance features
Dummy variables for categorical vars
Scale data
The general rule of thumb is that algorithms that exploit distances or similarities between data samples, such as artificial neural network (ANN), KNN, support vector machine (SVM), and k-means clustering, are sensitive to feature transformations. 
However, some tree-based algorithms such as decision tree (DT), RF, and GB are not affected by feature scaling. This is because tree-based models are not distance-based models and can easily handle varying range of features.
Feature normalization: Feature normalization guarantees that each feature will be scaled to [0,1] interval. 
Feature standardization (z-Score normalization): Standardization transforms each feature with Gaussian distribution to Gaussian distribution with a mean of 0 and a standard deviation of 1
In clustering algorithms such as k-means clustering, hierarchical clustering, density-based spatial clustering of applications with noise (DBSCAN), etc., standardization is important due to comparing feature similarities based on distance measures. On the other hand, normalization becomes more important with certain algorithms such as ANN and image processing 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)


Set train, validation and testing data
X = df.drop(['FlowPattern'], axis=1)
y = df['FlowPattern']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 10)


Model and pipeline
Parameter turning
It is difficult to know which combination of hyperparameters will work best based only on theory because there are complex interactions between hyperparameters. Hence the need for hyperparameter tuning: the only way to find the optimal hyperparameter values is to try many different combinations on a dataset.
LightGBM library faster than scikit one
Predictions
Confusin matrix and accuracy score
Predict on new dataset



