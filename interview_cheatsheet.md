# Interview Cheatsheet For Entry Level Data Analyst Jobs
##### Author: Emi Ly

##### Date: Feb 2022

##### [Tableau Dashboard] Coming Soon
#

*Overview of the concept from theory, statistics, machine learning model, SQL, and coding. I created this notebook to prepare for interview. Not all concept is covered. Source is everywhere from google search to class room lecture slides. I will update this constantly.*
### 💻 [Machine Learning](#machine-learning)
### 🔢 [Statistics](#statistics)
### 👩‍💻 [Coding](#coding)
### 📊 [SQL](#sql)
### 🎨 [Tableau](#tableau)
### 📺 [Business](#business)






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

[PCA](#pca)

#### Supervised vs Unsupervised
- Supervised: Input and output data are provided 
  - A supervised learning model produces an accurate result. It allows you to collect data or produce a data output from the previous experience. The drawback of this model is that decision boundaries might be overstrained if your training set doesn't have examples that you want to have in a class.
- Unsupervised: Input data are provided
  - In the case of images and videos, unsupervised algorithms can rapidly classify and cluster data using far fewer features than humans might specify, making data processing even faster and more efficient.
Unsupervised machine learning finds all kinds of unknown patterns in data. Also helps you to find features that can be useful for categorization. It is easier to get unlabeled data from a computer than labeled data, which needs manual intervention. Unsupervised learning solves the problem by learning the data and classifying it without any labels. 


#### Missing Values
- KNN Imputer: There are different ways to handle missing data. Some methods such as removing the entire observation if it has a missing value or replacing the missing values with mean, median, or mode values. However, these methods can waste valuable data or reduce the variability of your dataset. In contrast, KNN Imputer maintains the value and variability of your datasets, and yet it is more precise and efficient than using the average values: https://medium.com/@kyawsawhtoon/a-guide-to-knn-imputation-95e2dc496e
Needs to normalize data before applying KNN
- Fill na: `dataset['Some column']= dataset['Some column'].fillna(0)`
- Fill with mean: `dataset['some column']=dataset['some column'].fillna((dataset['some column'].mean()))`
- Fill with nearby value: `dataset['some column'] = dataset[‘some volumn'].fillna(method='ffill')`
- Don’t forget to `df_filling_mean.fillna(value = df_filling_mean.mean(), inplace = True)`
[Back to Top](#machine-learning)


#### Bias and Variance, Overfit and Underfit

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
[Back to Top](#machine-learning)
- 1. EDA on data
- 2. Detect outliers
- 3. Extract features
  - Use domain expertise
  - Feature ranking, selection
  - Feature collinearity
    - If the features are collinear, providing the model with the same information could potentially result in model confusion. Simply drop one of the collinear inputs. If both inputs are important to understand, it is advised to train two separate models with each collinear feature
   - Removing zero-variance features
- 4. Dummy variables for categorical vars
  - One hot encoding
- 5. Scale data
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


- 6. Set train, validation and testing data
```
X = df.drop(['FlowPattern'], axis=1)
y = df['FlowPattern']
```

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 10)
```

- 7. Model and pipeline
- 8. Parameter tuning
  - It is difficult to know which combination of hyperparameters will work best based only on theory because there are complex interactions between hyperparameters. Hence the need for hyperparameter tuning: the only way to find the optimal hyperparameter values is to try many different combinations on a dataset.
  - LightGBM library faster than scikit one
```
  param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

from sklearn.model_selection import GridSearchCV
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=5, n_jobs=-1)
nbModel_grid.fit(X_train, y_train)

print(nbModel_grid.best_estimator_)

y_pred = nbModel_grid.predict(X_test)
print(y_pred)
```
 
- 9. Predictions, Confusion matrix and accuracy score

- 10. Predict on new dataset
  - ![what to do after confusion matrix](https://user-images.githubusercontent.com/62857660/155050708-70d3312a-14e2-4710-8afa-64ca4f7bb23f.jpg)
[Back to Top](#machine-learning)

#### Feature Selection
[Back to Top](#machine-learning)
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

- Removing zero-variance features
```
from sklearn.feature_selection import VarianceThreshold
v_thres = VarianceThreshold(threshold=0)
v_thres.fit(X)
Cols = X.columns[v_thres.get_support()]
X = v_thres.transform(X)
X = pd.DataFrame(X,columns=Cols)
X
```

- To improve the model generalization (working with new data), in the feature selection process, those features that present a high collinearity with others were eliminated by means of a correlation matrix, taking a threshold value of 0.9 for this case (i.e., closely related features).
```
#Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

#Select the colums with high threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print('There are %d columns to remove.' % (len(to_drop)))
print(to_drop)

X = X.drop(to_drop, axis=1)
```


**Confusion Matrix**

[Back to Top](#machine-learning)
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
[Back to Top](#machine-learning)
- Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial.
 ```
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_knn))
score_knn = accuracy_score(y_test, y_pred_knn)
print("accuracy score: %0.3f" % score_knn)

```


#### Cross Validation
[Back to Top](#machine-learning)



#### KNN
[Back to Top](#machine-learning)

![Screenshot 2022-02-20 233754](https://user-images.githubusercontent.com/62857660/155051720-3fefa406-9fe6-4f67-8dbb-28b5edb16670.jpg)
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


#### Naive Bayes
[Back to Top](#machine-learning)

![Bayes-Theorem](https://user-images.githubusercontent.com/62857660/155053920-a5f71f97-98c3-401f-afc2-3770f14b2dde.png)
![download](https://user-images.githubusercontent.com/62857660/155053935-18afc22b-ecea-4aae-8188-0f84db6d1d89.png)

- It predicts membership probabilities for each class such as the probability that given record or data point belongs to a particular class.
- Typical applications include filtering spam, classifying documents, face recognition, weather prediction, news classification, medical diagnosis. It works well with high-dimensional data such as text classification, email spam detection.
- Naive Bayes is called naive because it assumes that each input variable is independent. This is a strong assumption and unrealistic for real data. In some cases, speed is preferred over higher accuracy.
- https://www.youtube.com/watch?v=l3dZ6ZNFjo0&t=28s
- 3 types of Naive Bayes models: Gaussian, Multinomial, and Bernoulli. 
  - Gaussian Naive Bayes – This is a variant of Naive Bayes which supports continuous values and has an assumption that each class is normally distributed. 
  - Multinomial Naive Bayes – Has features as vectors where sample(feature) represents frequencies with which certain events have occurred.
    - Multinomial naive Bayes assumes to have feature vector where each element represents the number of times it appears (or, very often, its frequency). ... The Gaussian Naive Bayes, instead, is based on a continuous distribution and it's suitable for more generic classification tasks.
```
GB = GaussianNB()
GB.fit(X_train, y_train)
predictions = GB.predict(X_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

#### PCA
```
from sklearn.decomposition import PCA
pca = PCA(n_components=3)

pc = pca.fit_transform(X)
```


## Statistics 
- Coming Soon
- A/B testing


## Coding
[Theory](#theory)

[Numpy](#numpy)

[Dataframe](#dataframe)

[loc vs iloc](#loc-vs-iloc)

[SNS](#sns)


#### Theory:
The primary difference between vectors and matrices is that vectors are 1d array while matrices are 2d arrays.


#### Numpy:

```
B =[[4,5,6],[7,8,9],[10,11,12]] 
np.array(B)
```
![numpy](https://user-images.githubusercontent.com/62857660/155055484-783dd1f1-bf38-466e-ac39-6893178033bb.jpg)


```
np.arange(0,40,5)
```
![numpy2](https://user-images.githubusercontent.com/62857660/155055537-a0bc1f60-2a74-42dc-aa6a-0a30298aed12.jpg)


```
A =np.arange(20,30) 
A.reshape(5,2)
```
![numpy3](https://user-images.githubusercontent.com/62857660/155055512-a7b25ff0-e349-4e4c-89ba-89381d7fd43a.jpg)


- The numpy random seed is a numerical value that generates a new set or repeats pseudo-random numbers. The value in the numpy random seed saves the state of randomness. If we call the seed function using value 1 multiple times, the computer displays the same random numbers.


#### Set, Tuple, Dictionary and List:
[Back to Top](#coding)

- Set is one of 4 built-in data types in Python used to store collections of data, the other 3 are List, Tuple, and Dictionary
- Set items are unordered, unchangeable, and do not allow duplicate values.


#### Dataframe:
[Back to Top](#coding)

- Create a dataframe: 
 ```
from numpy.random import randn
life_cycle = pd.DataFrame(randn(8,5),index = 'Land Seismic Geology Drilling Completions Production Facilities Midstream'.split(), columns = 'Cycle_1 Cycle_2 Cycle_3 Cycle_4 Cycle_5'.split())
```
- 8 rows, 5 columns
- Combine dataframe: `pd.concat()`
- Select columns: 
`life_cycle[['Cycle_2','Cycle_3','Cycle_4']]`

![list out column name](https://user-images.githubusercontent.com/62857660/155054706-5a0bccd3-c27e-41f1-9928-ad83a4f32fa9.jpg)

- If this drop is going to be permanent, please make sure to include `inplace = True`. To drop rows, use axis = 0 (which is the default in Python’s pandas) and to drop columns, use axis = 1 .
```life_cycle.drop(labels = ['Cycle_Total','Cycle_1_2_Mult'], axis=1, inplace=True)```


#### Loc vs iloc
[Back to Top](#coding)

- loc is label based
- iloc is index based

![loc iloc](https://user-images.githubusercontent.com/62857660/155054577-708bbdce-0b1c-4e69-8768-70374ec3552f.png)


- loc:

![loc](https://user-images.githubusercontent.com/62857660/155054621-b8b67eb6-1d91-474b-8f6f-7a7bd3fbf690.png)

- iloc:
```matrix.iloc[[0,2,4],[0,2,4]]```

![iloc](https://user-images.githubusercontent.com/62857660/155054741-624ee0ba-d287-4542-b669-51c845928858.jpg)



#### SNS
[Back to Top](#coding)

```sns.pairplot(df,hue ='CLASS')```

![snspair](https://user-images.githubusercontent.com/62857660/155054774-39e51323-07e2-4b8a-ba6e-f6ba09dbd104.png)

``` 
print(df['VAR1'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(df['VAR1'], color='g', bins=100, hist_kws={'alpha': 0.4});
```

![displot](https://user-images.githubusercontent.com/62857660/155054804-18a72861-60cf-4996-965d-a047cb210de8.jpg)

- SNS heatmap
```sns.heatmap(corr[(corr >= 0.01) | (corr <= -0.01)], 
 cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
  annot=True, annot_kws={"size": 10}, square=True);
```

![heatmap](https://user-images.githubusercontent.com/62857660/155054824-69356c18-ea92-4ff4-ac04-899b591d5e66.png)

## Coding
- Coming Soon

## SQL
- Coming Soon

## Tableau 
- Coming Soon

## Business
- Coming Soon
- KPI
- Project Management

[Back to Top](#machine-learning)
