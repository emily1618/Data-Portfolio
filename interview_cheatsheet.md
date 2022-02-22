# Interview Cheatsheet For Data Analyst Position
##### Author: Emi Ly

##### Date: Feb 2022

##### [Tableau Dashboard] Coming Soon
#

*Overview of the concept from theory, statistics, machine learning model, SQL, and coding. I created this notebook to prepare for interview. Source is from everywhere from google search to class room lecture slides.*
### 💻 [Machine Learning](#machine-learning)
### 🔢 [Statistics](#statistics)
### 👩‍💻 [Coding](#coding)
### 📊 [SQL](#sql)
### 🎨 [Tableau](#tableau)
### 📺[Learning Video Links](#learning-video-links)





## MACHINE LEARNING
[Supervised-vs-Unsupervised](#supervised-vs-unsupervised)

[Missing Values](#missing-values)

[Bias and Variance, Overfit and Underfit](#bias-and-variance-overfit-and-underfit)

[Dimension Reduction](#dimension-reduction)

[Flow](#flow)

[Feature Selection](#feature-selection)

[Confusion-Matrix](#confusion-matrix)

[F1 Score](#f1-score)

[Accuracy](#accuracy)

[Cross Validation](#cross-validation)

[KNN](#knn)

[Naive Bayes](#naive-bayes)

#### Supervised vs Unsupervised
- Supervised: Input and output data are provided 
  - A supervised learning model produces an accurate result. It allows you to collect data or produce a data output from the previous experience. The drawback of this model is that decision boundaries might be overstrained if your training set doesn't have examples that you want to have in a class.
- Unsupervised: Input data are provided
  - In the case of images and videos, unsupervised algorithms can rapidly classify and cluster data using far fewer features than humans might specify, making data processing even faster and more efficient.
Unsupervised machine learning finds all kinds of unknown patterns in data. Also helps you to find features that can be useful for categorization. It is easier to get unlabeled data from a computer than labeled data, which needs manual intervention. Unsupervised learning solves the problem by learning the data and classifying it without any labels. 


#### Missing Values
- KNN Imputer: There are different ways to handle missing data. Some methods such as removing the entire observation if it has a missing value or replacing the missing values with mean, median, or mode values. However, these methods can waste valuable data or reduce the variability of your dataset. In contrast, KNN Imputer maintains the value and variability of your datasets, and yet it is more precise and efficient than using the average values: https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e
Needs to normalize data before applying KNN
- Fill na: dataset['Some column']= dataset['Some column'].fillna(0)
- Fill with mean: dataset['some column']=dataset['some column'].fillna((dataset['some column'].mean()))
- Fill with nearby value: dataset['some column'] = dataset[‘some volumn'].fillna(method='ffill')
- Don’t forget to df_filling_mean.fillna(value = df_filling_mean.mean(), inplace = True)


#### Bias and Variance, Overfit and Underfit
[Back to Top](#author-emi-ly)
- The inability of a ML model to capture the true relationship is called “bias.” Models with high bias are unable to capture the true relationship between input and output features, and it usually leads to oversimplification of the model.
Under fit
  - An underfit model has high bias and low variance.
- A model with high variance means it fits the training data very well but does a poor job predicting the testing data. It other words, it memorizes the training data very well and is not able to predict the test data due to low generalization.
  - Over fit
  - An overfit model means it has high variance and low bias.
  - ![fit](https://user-images.githubusercontent.com/62857660/155050192-fa6ff06c-5271-43a9-8054-cdb5464b0404.jpg)


#### Dimension Reduction
- Dimensionality reduction is the process of reducing the number of variables by obtaining a set of important variables.
- PCA 



#### Flow
- EDA on data
- Detect outliers
- Extract features
  - Use domain expertise
  - Feature ranking, selection
  - Feature collinearity
    - If the features are collinear, providing the model with the same information could potentially result in model confusion. Simply drop one of the collinear inputs. If both inputs are important to understand, it is advised to train two separate models with each collinear feature
   - Removing zero-variance features
- Dummy variables for categorical vars
- Scale data
  - The general rule of thumb is that algorithms that exploit distances or similarities between data samples, such as artificial neural network (ANN), KNN, support vector machine (SVM), and k-means clustering, are sensitive to feature transformations. 
  - However, some tree-based algorithms such as decision tree (DT), RF, and GB are not affected by feature scaling. This is because tree-based models are not distance-based models and can easily handle varying range of features.
  - Feature normalization: Feature normalization guarantees that each feature will be scaled to [0,1] interval. 
  - Feature standardization (z-Score normalization): Standardization transforms each feature with Gaussian distribution to Gaussian distribution with a mean of 0 and a standard deviation of 1
  - In clustering algorithms such as k-means clustering, hierarchical clustering, density-based spatial clustering of applications with noise (DBSCAN), etc., standardization is important due to comparing feature similarities based on distance measures. On the other hand, normalization becomes more important with certain algorithms such as ANN and image processing .
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)
```


- Set train, validation and testing data
```
X = df.drop(['FlowPattern'], axis=1)
y = df['FlowPattern']
```
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 10)
```


- Model and pipeline
- Parameter tuning
  - It is difficult to know which combination of hyperparameters will work best based only on theory because there are complex interactions between hyperparameters. Hence the need for hyperparameter tuning: the only way to find the optimal hyperparameter values is to try many different combinations on a dataset.
  - LightGBM library faster than scikit one
 
- Predictions, Confusion matrix and accuracy score

- Predict on new dataset
  - ![what to do after confusion matrix](https://user-images.githubusercontent.com/62857660/155050708-70d3312a-14e2-4710-8afa-64ca4f7bb23f.jpg)


#### Feature Selection
![feature](https://user-images.githubusercontent.com/62857660/155051328-8c9f20ce-3beb-4fa0-88bf-940ee2fa52b1.jpg)


`
import lightgbm as lgb
`

- Find the features with zero importance
```
zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
print('There are %d features with 0.0 importance' % len(zero_features))
```
- Well, it also looks like many of the features we made have literally 0 importance. For the gradient boosting machine, features with 0 importance are not used at all to make any splits. Therefore, we can remove these features from the model with no effect on performance (except for faster training).
```
Xfi_Train = X_Train.drop(columns = zero_features)
Xfi_Test = X_Test.drop(columns = zero_features)
```


**Confusion Matrix**
- Each row of the matrix represents the instances in a predicted class,.
- Each column represents the instances in an actual class.
- ![confusion](https://user-images.githubusercontent.com/62857660/155051392-1d96187e-70d7-418f-8f19-cc2d91e7f3ab.jpg)

#### F1 Score
`
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))
`
![classification](https://user-images.githubusercontent.com/62857660/155051557-02149529-09a5-4ca6-b71b-4e2ad00584a8.jpg)

- average returns the average without considering the proportion for each label in the dataset. weighted returns the average considering the proportion for each label in the dataset.
-  Precision (tp / (tp + fp) ) measures the ability of a classifier to identify only the correct instances for each class.
-  Recall (tp / (tp + fn) is the ability of a classifier to find all correct instances per class. 
F score of 1 indicates a perfect balance as precision and the recall are inversely related. A high F1 score is useful where both high recall and precision is important. 
- Support is the number of actual occurrences of the class in the test data set. Imbalanced support in the training data may indicate the need for stratified sampling or rebalancing.

#### Accuracy
- Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial.
 ```
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_knn))
score_knn = accuracy_score(y_test, y_pred_knn)
print("accuracy score: %0.3f" % score_knn)
```

#### Cross Validation

#### KNN
- ![Screenshot 2022-02-20 233754](https://user-images.githubusercontent.com/62857660/155051720-3fefa406-9fe6-4f67-8dbb-28b5edb16670.jpg)
- Supervised
- Based on feature similarity
- The algorithm can be used to solve both classification and regression problem 
- The advantage of nearest-neighbor classification is its simplicity. There are only two choices a user must make: (1) the number of neighbors, k, and (2) the distance metric to be used. Common choices of distance metrics include Euclidean distance and city-block Manhattan distance.
- K-NN is a lazy learner because it doesn't learn a discriminative function from the training data but “memorizes” the training dataset instead. 
- KNN works well with smaller datasets because it is a lazy learner. It needs to store all the data and then make a decision only at run time. So if the dataset is large, there will be a lot of processing which may adversely impact the performance of the algorithm. Each input variable can be considered a dimension of a p-dimensional input space. In high dimensions, points that may be similar may have very large distances.
- It is advised to use the KNN algorithm for multiclass classification if the number of samples of the data is less than 50,000. 
- How we choose k: parameter tuning. Improve accuracy. We can use the square root of n (number of samples) or choose an odd value of k.
- Sample: https://www.youtube.com/watch?v=4HKqjENq9OU
 ```
from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)
 ```
 ```
# Testing the model using X_test and storing the output in y_pred
y_pred_knn = knn.predict(X_test_scaled)
y_pred_knn
```



