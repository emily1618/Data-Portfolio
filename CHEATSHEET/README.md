# Interview Cheatsheet For Entry Level Data Analyst Jobs
#### I created this notebook to prepare for an interview. The information is an overview of the concept from theory, statistics, machine learning model, SQL, and coding. Not all concept is covered. The source is everywhere from google search to classroom lecture slides. I will continually update this. 

##### Author: Emi Ly

##### Date: Feb 2022
#

### 💻 [Machine Learning](#machine-learning)
### 🔢 [Statistics](#statistics)
### 👩‍💻 [Coding](#coding)
### 📊 [SQL](#sql)
### 📉 [Excel](#excel)
### 🎨 [Tableau + Power BI](#tableau-and-power-bi)
### 📺 [Business](#business)






## MACHINE LEARNING
An application of Artificial Intelligence wherein the systems get the ability to automatically learn and improve based on experience (past data) 
- Classification. yes or no
  - naive bayyes, logistic regression - for simpler data
  - decision tree, random forest - for complicated data. large data use random forest
- Regression. predict price
- Clustering. product recommendation
- Source:https://www.youtube.com/watch?v=RmajweUFKvM
- Check run time using
```
- t0 = time()
< your clf.fit() line of code >
print "training time:", round(time()-t0, 3), "s"
```

[Supervised-vs-Unsupervised](#supervised-vs-unsupervised)

[Sample Size](#sample-size)

[Entropy and Resampling](#resampling)

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

[Random Forest](#random-forest)

[Decision Tree](#decision-tree)

[Support Vector Machine](#support-vector-machine)

#### Supervised vs Unsupervised
- Supervised: Input and output data are provided 
  - A supervised learning model produces an accurate result. It allows you to collect data or produce a data output from the previous experience. The drawback of this model is that decision boundaries might be overstrained if your training set doesn't have examples that you want to have in a class. Usage: bank loan approval
- Unsupervised: Input data are provided. Group like things together:
  - In the case of images and videos, unsupervised algorithms can rapidly classify and cluster data using far fewer features than humans might specify, making data processing even faster and more efficient.
Unsupervised machine learning finds all kinds of unknown patterns in data. Also helps you to find features that can be useful for categorization. It is easier to get unlabeled data from a computer than labeled data, which needs manual intervention. Unsupervised learning solves the problem by learning the data and classifying it without any labels. 
- Reinforcement: Input data one at a time. Machine learning has to adjust accordingly. 

#### Sample Size
More data >  Fine tune model

![Screenshot 2022-03-27 135751](https://user-images.githubusercontent.com/62857660/160296481-2b5adbc7-a93d-4134-9f19-a9172c7daa3b.png)


#### Resampling
- Resampling methods are very useful and beneficial in statistics and machine learning to fit more accurate models, model selection, and parameter tuning. Involve repeatedly drawing samples from a dataset and calculating statistics and metrics on each of those samples in order to obtain further information. Use `from sklearn.utils import resample`
- Entropy: the measure of randomness in the dataset
- ![entropy](https://user-images.githubusercontent.com/62857660/156621431-6e07b9df-1368-48f5-b8ca-5baad6fda963.JPG)


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
Over fit
  - An overfit model means it has high variance (unstable due to small variation) and low bias (hard to work with new data).
  - Capture too much noise. Solving for one specific incident instead of the general solution
  - Checking for overfitting. If the score is close, may not be a case of overfitting
  - ```print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))```
  - ![fit](https://user-images.githubusercontent.com/62857660/155050192-fa6ff06c-5271-43a9-8054-cdb5464b0404.jpg)
 
 - This is an example of the bias-variance tradeoff in machine learning. A model with high variance has learned the training data very well but often cannot generalize to new points in the test set. On the other hand, a model with high bias has not learned the training data very well because it does not have enough complexity. This model will also not perform well on new points.

- VIdeo explanation: https://www.youtube.com/watch?time_continue=69&v=W5uUYnSHDhM&feature=emb_logo

#### Dimension Reduction
- Dimensionality reduction is the process of reducing the number of variables by obtaining a set of important variables.
- PCA 


#### Flow
[Back to Top](#machine-learning)
- 1 EDA on data
- 2 Detect outliers
- 3 Extract features
  - Use domain expertise
  - Feature ranking, selection
    - LightGBM: a gradient boosting framework that uses tree-based learning algorithm.
  - Feature collinearity
    - If the features are collinear, providing the model with the same information could potentially result in model confusion. Simply drop one of the collinear inputs. If both inputs are important to understand, it is advised to train two separate models with each collinear feature
   - Removing zero-variance features
- 4 Dummy variables for categorical vars
  - One hot encoding
- 5 Scale data
  - The general rule of thumb is that algorithms that exploit distances or similarities between data samples, such as artificial neural network (ANN), KNN, support vector machine (SVM), and k-means clustering, are sensitive to feature transformations. 
  - However, some tree-based algorithms such as decision tree (DT), RF, and GB are not affected by feature scaling. This is because tree-based models are not distance-based models and can easily handle a varying range of features.
  - Feature normalization: Feature normalization guarantees that each feature will be scaled to [0,1] interval. 
  - Feature standardization (z-Score normalization): Standardization transforms each feature with Gaussian distribution to Gaussian distribution with a mean of 0 and a standard deviation of 1
  - In clustering algorithms such as k-means clustering, hierarchical clustering, density-based spatial clustering of applications with noise (DBSCAN), etc., standardization is important due to comparing feature similarities based on distance measures. On the other hand, normalization becomes more important with certain algorithms such as ANN and image processing .
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)
```


- 6 Set train, validation and testing data
```
X = df.drop(['FlowPattern'], axis=1)
y = df['FlowPattern']
```

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 10)
```

- 7 Model and pipeline
- 8 Parameter tuning
  - It is difficult to know which combination of hyperparameters will work best based only on theory because there are complex interactions between hyperparameters. Hence the need for hyperparameter tuning: the only way to find the optimal hyperparameter values is to try many different combinations on a dataset.
  - After spending hours cleaning the data to fit your model and tuning the parameters using GridSearchCV, you may come to find that all that hyper tuning didn’t improve your model performance very much. If you took a whole day to test out parameters and only improved your model accuracy by 0.5%, perhaps that wasn’t the best use of your time.
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


- 9 Confusion matrix and accuracy score
Score for training data: `print(f'Model Accuracy: {tree.score(X_train, y_train)}')`

Score for testing data: `score_tree = accuracy_score(y_test, y_pred2)*100`

- 10 Predict on new dataset
  - ![what to do after confusion matrix](https://user-images.githubusercontent.com/62857660/155050708-70d3312a-14e2-4710-8afa-64ca4f7bb23f.jpg)
[Back to Top](#machine-learning)

#### Feature Selection
[Back to Top](#machine-learning)

- Better accuracy.
- Capable of handling large-scale data. It is not advisable to use LGBM on small datasets. Light GBM is sensitive to overfitting and can easily overfit small data.
- https://www.kaggle.com/prashant111/lightgbm-classifier-in-python
- Although `GridSearchCV` has numerous benefits, you may not want to spend too much time and effort perfectly tuning your model. Better use of time may be to investigate your features further. Feature engineering and selecting subsets of features can increase (or decrease) the performance of your model tremendously. 

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

Source: https://www.youtube.com/watch?v=prWyZhcktn4&list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy&index=17

Precision: how often is it right. TP/(TP+FP)
Recall: How often doe the model predict the correct positive values. TP/(TP+FN)

#### F1 Score

`
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_knn))
`
![classification](https://user-images.githubusercontent.com/62857660/155051557-02149529-09a5-4ca6-b71b-4e2ad00584a8.jpg)

- average returns the average without considering the proportion for each label in the dataset. weighted returns the average considering the proportion for each label in the dataset.
-  Precision (tp / (tp + fp) ) measures the ability of a classifier to identify only the correct instances for each class.
-  Recall (tp / (tp + fn) is the ability of a classifier to find all correct instances per class. 
F score of 1 indicates a perfect balance as precision and recall are inversely related. A high F1 score is useful where both high recall and precision is important. 
- Support is the number of actual occurrences of the class in the test data set. Imbalanced support in the training data may indicate the need for stratified sampling or rebalancing.


#### Accuracy
[Back to Top](#machine-learning)

How often is our classifier is right: sum of all true values divided by total values

- Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial.
 ```
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_knn))
score_knn = accuracy_score(y_test, y_pred_knn)
print("accuracy score: %0.3f" % score_knn)

```
Score for training data: `print(f'Model Accuracy: {tree.score(X_train, y_train)}')`

Score for testing data: `score_tree = accuracy_score(y_test, y_pred2)*100`


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
- KNN works well with smaller datasets because it is a lazy learner. It needs to store all the data and then make a decision only at run time. So if the dataset is large, there will be a lot of processing which may adversely impact the performance of the algorithm. Each input variable can be considered a dimension of p-dimensional input space. In high dimensions, points that may be similar may have very large distances.
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

- It predicts membership probabilities for each class such as the probability that a given record or data point belongs to a particular class.
- Typical applications include filtering spam, classifying documents, face recognition, weather prediction, news classification, medical diagnosis. It works well with high-dimensional data such as text classification, email spam detection.
- Naive Bayes is called naive because it assumes that each input variable is independent. This is a strong assumption and unrealistic for real data. In some cases, speed is preferred over higher accuracy.
- https://www.youtube.com/watch?v=l3dZ6ZNFjo0&t=28s
- 3 types of Naive Bayes models: Gaussian, Multinomial, and Bernoulli. 
  - Gaussian Naive Bayes – This is a variant of Naive Bayes which supports continuous values and has an assumption that each class is normally distributed. 
  - Multinomial Naive Bayes – Has features as vectors where sample(feature) represents frequencies with which certain events have occurred.
    - Multinomial naive Bayes assumes to have a feature vector where each element represents the number of times it appears (or, very often, its frequency). ... The Gaussian Naive Bayes, instead, is based on a continuous distribution and it's suitable for more generic classification tasks.
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

#### Random Forest
Limiting the depth of a single decision tree is one way we can try to make a less biased model. Another option is to use an entire forest of trees, training each one on a random subsample of the training data. The final model then takes an average of all the individual decision trees to arrive at a classification. This is the idea behind the random forest.

#### Decision Tree
Each branch of the tree represents a possible decision. Can be used on classification and regression. Reduce entropy and calculate the gain. 
Entropy controls how a DT decides where to split. Entropy measures the impurity in the dataset.
Entropy = -Sum(Pi(log2)(Pi)) or 
```
import scipy.stats
print scipy.stats.entropy([2,1],base=2)
```

![Screenshot 2022-03-26 234252](https://user-images.githubusercontent.com/62857660/160267198-2bf5ad2d-d73e-48fb-83d6-01d4ff65fe44.png)
![Screenshot 2022-03-26 235136](https://user-images.githubusercontent.com/62857660/160267200-e653b7c5-c394-442e-a323-4d0b8b094e84.png)


Information gain:
Entropy(parent) - weight average Entropy(children)
If information gain = 0, then not a good idea to split on that feature first. 
Perfert purity = 1 
https://www.youtube.com/watch?v=o75xNa_jwvg&feature=emb_logo

![node](https://user-images.githubusercontent.com/62857660/156621764-1048de9b-f744-4d98-bed3-bf8b37e54be3.JPG)

Good: Simple to understand and data prep, able to handle numerical and categorical, non-linear parameter don't affect performance
Bad: Can be overfit

![Screenshot 2022-03-26 233426](https://user-images.githubusercontent.com/62857660/160266800-354687ad-4543-4e7d-bc99-02d07badc5c2.png)
![Screenshot 2022-03-26 233822](https://user-images.githubusercontent.com/62857660/160266889-5398f444-ca1e-4c0f-899e-4007fd0826ea.png)


#### Support Vector Machine
Use in ace dectection, bioinformatics, images classification.
Supervised for classfication and regression.
Using kernel to transform 1D to 2D to 3D
Advantage: high dimensional space (tokenzied word in doc) and naturally avoid overfitting
 ![Screenshot 2022-03-26 212932](https://user-images.githubusercontent.com/62857660/160264467-377a1ed9-1f6d-4037-98c3-f3de0f0a9db4.png)![Screenshot 2022-03-26 213039](https://user-images.githubusercontent.com/62857660/160264469-a48b2e8c-294f-4522-9cec-a8913b7191ce.png)

![Screenshot 2022-03-26 224918](https://user-images.githubusercontent.com/62857660/160265828-41d44aef-f624-4671-9f56-bead9c760703.png)
![Screenshot 2022-03-26 230243](https://user-images.githubusercontent.com/62857660/160266105-ed6e89ca-c1f7-45f0-9b50-9d0be5f6541a.png)

![Screenshot 2022-03-26 230613](https://user-images.githubusercontent.com/62857660/160266797-41717b99-deff-4895-822e-7d1843c7690e.png)


[Source](https://www.youtube.com/watch?v=TtKF996oEl8)

### Which one to choose?
SVM might be slow if big dataset with lots of noise - https://www.youtube.com/watch?v=U9-ZsbaaGAs&t=47s
Navies Bayes will be faster than SVM


## Statistics 
- Power: the higher the statistical power, the better the test is. Use in min sample size experiment
	- Person infected with covid and test show positive, power of test
- Type 1 Error: when we reject the true null hypothsis, conclude the findings are significant but in fact they occurred by chance. In A/B testing, observed difference when there is no different. 
	- Person is not infected but test show positive
- Type 2 Error: fail to reject the false null, conclude the findings are NOT significant but in fact they are. In A/B, observed no difference but there is difference.
	- Person is infected but test show neegative
- Confidence Interval: how variable the sample may be, an estimate of the true value (how often it will contain the true value). The probability that it will cover the true value is CI.
	- Level of uncertainty when we estimate a value
	- 95% of CI avg of height is 168 to 180cm. How likely? We're 95% confident that the range cover the true avg height. 
- P-value: <0.5, reject the null. p>0.05, cannot reject null. in A/B, the smaller the p-value, more convinced there is a difference b/w the control and treatment.
	- Can we conclude avg height is 175cm? p-value measure what data we observe and what conclussion we can draw. If p<0.05, we reject the avg ht is 175cm, 
- Examples2
- A/B testing
	1. Hypothesis: By creating a new feature or improve a new existing feature, we want to increase the number of users by X percentage. Goal is to grow the user. The X percentage is a practical significance (bound and min detectable effect). If we did not able to reach the bound, then we have to decide if we want to launch this feature. 
	2. Define a metric, success metrics vs guardrail metrics
	3. Power Analysis - alpha, beta and delta to calculate sample size. Number of days/weeks to run the experiment
	4. Ramp up the experiment - make sure the data we use is good enough
	5. Interpret results - problems such as control group vs treatment group does not have statisical signifance
	- Problems occurred other than looking at the data or p-value: most importantly, we want to make sure the data is trusthworthy! Second, we can look at segmentation. Third, funnel analysis. Forth, decomposition analysis (eg: daily active user can futher divide into new users or old users). Fifth, selection bias. 
- Metrics
	1. Primary metric: when you add/change a feature, it will directly impact your users. Eg: click through rate when increase a checkout button size
	2. Secondary metric: same as primary, but not as important as primary, will assist along with primary. Eg: user checkout time reduced when button size increase
	3. Guardrail metric: we want that the experiment do not affect this metric. Eg: avg sales price should not be affect bu the checkout button size increase
- Variance 
  - how to spread out the dataset is
  - we need a variance to make predictions
- Std
  - low std, close to the mean
  - high std, far to the mean


## Coding
[General](#general)

[Pandas](#pandas)

[Numpy](#numpy)

[Dataframe](#dataframe)

[loc vs iloc](#loc-vs-iloc)

[SNS](#sns)


#### General:
The primary difference between vectors and matrices is that vectors are 1d arrays while matrices are 2d arrays.

- lambda function: perform a function on every row of the dataset

`day_mapper = {0: "Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}`


`data_import["DayOfWeek"] = data_import["Date"].map(lambda x: day_mapper[x.dayofweek])`


- using `~`, if the data is in the conditions, using ~ get rid of those data
`
full_calendar = [~full_calendar["Weekday"].isin(["Saturday","Sunday"])]
`

- `.to_clipboard()` takes the dataframe and copy/paste to your excel



#### Pandas:
`pd.concat` to join columns

`print(f'Feature Columns: {", ".join(features)}\nLabel Column: {labels}')`
![cc](https://user-images.githubusercontent.com/62857660/156618078-9914bb93-9ac8-47ca-93ff-906b91d052ae.JPG)




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

If this drop is going to be permanent, please make sure to include `inplace = True`. To drop rows, use axis = 0 (which is the default in Python’s pandas) and to drop columns, use axis = 1 .
```life_cycle.drop(labels = ['Cycle_Total','Cycle_1_2_Mult'], axis=1, inplace=True)```

Sum numbers in a column and skip the NaN
```
df["sum_a_and_b"] = df[["a", "b"]].sum(axis=1)
```


#### Loc vs iloc
[Back to Top](#coding)

loc is label based
iloc is index-based

![loc iloc](https://user-images.githubusercontent.com/62857660/155054577-708bbdce-0b1c-4e69-8768-70374ec3552f.png)


loc:

![loc](https://user-images.githubusercontent.com/62857660/155054621-b8b67eb6-1d91-474b-8f6f-7a7bd3fbf690.png)

iloc:
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

```
fig = plt.figure(figsize=(10,4))

ax1 = fig.add_subplot(121)
ax1.hist(df['Birth Year'], bins=20, color='g')
ax1.set_title('hits x users histogram')

dfxx = df['Birth Year'].describe().to_frame().round(2)

ax2 = fig.add_subplot(122)
font_size=12
bbox=[0, 0, 1, 1]
ax2.axis('off')
mpl_table = ax2.table(cellText = dfxx.values, rowLabels = dfxx.index, bbox=bbox, colLabels=dfxx.columns)
mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(font_size)
```
![download](https://user-images.githubusercontent.com/62857660/155908924-943d26f0-f03e-49d2-9a5f-85653052f72f.png)



- SNS heatmap
```sns.heatmap(corr[(corr >= 0.01) | (corr <= -0.01)], 
 cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
  annot=True, annot_kws={"size": 10}, square=True);
```

![heatmap](https://user-images.githubusercontent.com/62857660/155054824-69356c18-ea92-4ff4-ac04-899b591d5e66.png)

- SNS for df.describe()
```
num_col = df._get_numeric_data().columns

describe_num_df = df.describe(include=['int64','float64'])
describe_num_df.reset_index(inplace=True)
# To remove any variable from plot
describe_num_df = describe_num_df[describe_num_df['index'] != 'count']
for i in num_col:
  if i in ['index']:
    continue
  sns.factorplot(x='index', y=i, data=describe_num_df, figsize=(25, 10))
  plt.show()
```

## Coding
- Coming Soon

## SQL
Ensure data integrity: only the data you want to be entered is entered, and only certain users are able to enter data into the database.

Ensure domains: a set of values following documentations, consistent data types

Normalization is important so the data can follow logical routes and queries efficiently. Without normalization, database systems can be inaccurate, slow, and inefficient. Four GOALS:
  -  arranging data into logical groupings such that each group describes a small part of the whole;
  -  minimizing the amount of duplicate data stored in a database;
  -  organizing the data such that, when you modify it, you make the change in only one place;
  -  building a database in which you can access and manipulate the data quickly and efficiently without compromising the integrity of the data.

AVG ignores the NULL values

DISTINCT to group by columns without aggregation

LIKE and ILIKE(Protegre SQL). ILIKE is not case sensitive

WHERE and HAVING
```
GROUP BY a.id, a.name, w.channel
HAVING COUNT(*) > 6 AND w.channel = 'facebook'
ORDER BY use_of_channel;
```
same as

```
where w.channel = 'facebook'
group by a.name, w.channel
having count(w.id) > 6
order by channel_count;
```

USE UPPER if the result have both upper and lower, but you want upper instead
```
select left(upper(name), 1) as first_letter, count(*) as distribution
from accounts
group by first_letter
order by first_letter;
```

Split first and last name if seperate by space
```
select name,
       left(name, position(' ' in name)) as first,
       right(name, length(name) - position(' ' in name)) as last
from accounts;
```
**OR**
```
SELECT  name,
	LEFT(name, STRPOS(name, ' ')-1)  first, 
	RIGHT(name, LENGTH(name) - STRPOS(name, ' ')) last
 FROM accounts;
 ```

Remove space by replacing a space with no space `replace(name,' ','')`
```
select primary_poc, 
      concat(first,'.',last,'@',lower(replace(name,' ','')),'.com') as email
from (select primary_poc,name,
       left(primary_poc, position(' ' in primary_poc)) as first,
       right(primary_poc, length(primary_poc) - position(' ' in primary_poc)) as last
      from accounts) sub;
```
![Screenshot 2022-03-21 224358](https://user-images.githubusercontent.com/62857660/159403650-6091dfed-4943-453b-8171-c3d122ea88bd.png)



USE MIN and MAX for the date range
 - When was the earliest order ever placed? `SELECT MIN(occurred_at)`
 - hen did the most recent (latest) web_event occur? `SELECT MAX(occurred_at`
 
SELECT MIN(occurred_at) 
FROM orders;
LIKE This allows you to perform operations similar to using WHERE and =, but for cases when you might not know exactly what you are looking for.
  -  frequently used with %
  -  % is a wildcard, eg: `WHERE name LIKE %google%` is any characters, then contain the text 'google', then any characters after
  -  if name column is first name space last name, to get the last name start with any letter, use `WHERE table.name_column LIKE '% A%'`
  
IN This allows you to perform operations similar to using WHERE and =, but for more than one condition.
  - like OR, but cleaner
  - `WHERE name IN ('Walmart', 'Target', 'Nordstrom')`
  
NOT This is used with IN and LIKE to select all of the rows NOT LIKE or NOT IN a certain condition.
`WHERE column BETWEEN 6 AND 10` is a cleaner version of `WHERE column >= 6 AND column <= 10`
  - BETWEEN endpoint values are included, but tricky for dates. To get all orders in 2016, use `occurred_at BETWEEN '2016-01-01' AND '2017-01-01'`
  
Make sure to include parenthesis
  - `select name from accounts
     where name like 'C%' Or name like 'W%' AND (primary_poc like '%ana%' OR primary_poc like '%Ana%') AND (primary_poc not like '%eana');` 
  - ```SELECT * FROM accounts
      WHERE (name LIKE 'C%' OR name LIKE 'W%') 
           AND ((primary_poc LIKE '%ana%' OR primary_poc LIKE '%Ana%') 
           AND primary_poc NOT LIKE '%eana%');```
  - ABOVE IS NOT THE SAME!!!

INNER join result may be the same if flip the table on a LEFT join.

If you have two or more columns in your SELECT that have the same name after the table name such as accounts.name and sales_reps.name you will need to alias them. Otherwise, it will only show one of the columns. You can alias them like accounts.name AS AcountName, sales_rep.name AS SalesRepName

UNION and UNION ALL, CROSS JOIN, and the tricky SELF JOIN.

DATE is databases are in YYYY-MM-DD 00:00:00
![1](https://user-images.githubusercontent.com/62857660/158932927-464bdeff-5266-4ef4-a255-a8fec70a0d04.png)

![2](https://user-images.githubusercontent.com/62857660/158932933-a512fc05-36b6-4fdf-8782-2a53b31c96e3.png)

CASE statement can in the SELECT phrase

If you use aggregate, you must put them in a GROUP BY. You can also group by case statement columns
```
select  
   CASE WHEN total >= 2000 THEN 'At least 2000'
   	    When total Between 1000 and 2000 then 'Between 1000 and 2000'
            ELSE 'Less and 1000' END AS order_level,
     count(*) order_count
FROM orders
group by order_level;
```
![Screenshot 2022-03-17 223456](https://user-images.githubusercontent.com/62857660/158932909-9c0a7b95-7abc-48bd-ade4-7c4552a69fcf.png)

Identify top sales rep by # of orders
```
select s.name as sales_name, count(o.id) as orders, 
		case when count(o.id) > 200 then 'top'
        else 'not' end as level
from sales_reps s
join accounts a
on s.id = a.sales_rep_id
join orders o
on a.id = o.account_id
group by s.name
order by count(o.id) desc;
```
![Screenshot 2022-03-17 230905](https://user-images.githubusercontent.com/62857660/158935889-745cf705-50c1-4408-b3a8-832c3b5dcf8b.png)


Top sales rep based on total sales
```
SELECT s.name, COUNT(*), SUM(o.total_amt_usd) total_spent, 
     CASE WHEN COUNT(*) > 200 OR SUM(o.total_amt_usd) > 750000 THEN 'top'
     WHEN COUNT(*) > 150 OR SUM(o.total_amt_usd) > 500000 THEN 'middle'
     ELSE 'low' END AS sales_rep_level
FROM orders o
JOIN accounts a
ON o.account_id = a.id 
JOIN sales_reps s
ON s.id = a.sales_rep_id
GROUP BY s.name
ORDER BY 3 DESC;
```
![Screenshot 2022-03-17 231453](https://user-images.githubusercontent.com/62857660/158936380-3f94dad0-cb0d-463a-a7f9-285120a0526e.png)

Subquery
If you are only returning a single value, you might use that value in a logical statement like WHERE, HAVING, or even SELECT - the value could be nested within a CASE statement. You should not include an alias when you write a subquery in a conditional statement. This is because the subquery is treated as an individual value (or set of values in the IN case) rather than as a table.
```
SELECT AVG(standard_qty) avg_std, AVG(gloss_qty) avg_gls, AVG(poster_qty) avg_pst, SUM(total_amt_usd)
FROM orders
WHERE DATE_TRUNC('month', occurred_at) = 
     (SELECT DATE_TRUNC('month', MIN(occurred_at)) FROM orders);
```
![12](https://user-images.githubusercontent.com/62857660/159071589-ff229023-16bc-429f-9ccd-a8da8f3e5eea.JPG)

Sometimes, you will need to join a table with a subquery table and you can use CTE
```
WITH table1 AS (
          SELECT *
          FROM web_events),

     table2 AS (
          SELECT *
          FROM accounts)
SELECT *
FROM table1
JOIN table2
ON table1.account_id = table2.id;
```

Coalesce with an anther id
```
SELECT COALESCE(o.id, a.id) filled_id, a.name, a.website, a.lat, a.long, a.primary_poc, a.sales_rep_id, o.*
FROM accounts a
LEFT JOIN orders o
ON a.id = o.account_id
WHERE o.total IS NULL;
```

A window function performs a calculation across a set of table rows that are somehow related to the current row. But unlike regular aggregate functions, use of a `window` function does not cause rows to become grouped into a single output row — the rows retain their separate identities. You can’t include window functions in a GROUP BY clause. Window functions are permitted only in the SELECT list and the ORDER BY clause of the query. They are forbidden elsewhere, such as in GROUP BY, HAVING and WHERE clauses.

```
SELECT standard_amt_usd,
       date_trunc('year', occurred_at) as year,
       SUM(standard_amt_usd) OVER (partition by date_trunc('year', occurred_at)  ORDER BY  occurred_at) AS running_total
FROM orders;
```
![Screenshot 2022-03-22 235221](https://user-images.githubusercontent.com/62857660/159626513-5a11744e-f856-4fec-b61e-354b04f2ad07.png)

Partition by
![Screenshot 2022-03-23 083949](https://user-images.githubusercontent.com/62857660/159712820-dba55540-3ee5-429f-9e85-9c912424fe58.png)

Row() and Rank() difference is when two lines in a row have the same values, then Row() will give different number and Rank() will give the same number.
Dense_Rank doesn't not skip the values, but Rank() skip it

```
SELECT id,
       account_id,
       total,
       RANK() OVER (PARTITION BY account_id ORDER BY total DESC) AS total_rank
FROM orders
```
![Screenshot 2022-03-23 085015](https://user-images.githubusercontent.com/62857660/159714953-fa5edef2-df80-4075-aa72-d5d6eb4f7087.png)

`Dense_Rank()`
![dense](https://user-images.githubusercontent.com/62857660/159749834-013d079b-34d6-4779-857d-02300e8555fc.PNG)

If you wanted to return unmatched rows only, which is useful for some cases of data assessment, you can isolate them by adding the following line to the end of the query: `WHERE Table_A.column_name IS NULL OR Table_B.column_name IS NULL`. Examples usage are: I've used full outer joins when attempting to find mismatched, orphaned data, from both of my tables and wanted all of my result set, not just matches.
![full-outer-join-if-null](https://user-images.githubusercontent.com/62857660/159950748-d3736545-4ab9-4952-9237-638035c7750d.png)

Details of `UNION` 
- There must be the same number of expressions in both SELECT statements.
- The corresponding expressions must have the same data type in the SELECT statements. For example: expression1 must be the same data type in both the first and second 
SELECT statement.

Expert Tip
- UNION removes duplicate rows.
- UNION ALL does not remove duplicate rows. **use more in real life**

#

#

## Excel
- VLOOKUP always start with the left-most column, using INDEX and MATCH instead
- `VLOOKUP(value to look for, range to look in, column number of the value to return, approximate or exact match [TRUE/FALSE])`
- `=INDEX(array or reference, MATCH(value to look up,lookup_array,[match_type]) `
- `=VLOOKUP(A17,C2:H14,4,FALSE)` will have the same result as `=INDEX(F2:F14, MATCH(A17,C2:C14,0))`
  - index, match, the match is a powerful tool
![Capture](https://user-images.githubusercontent.com/62857660/156065004-5b03b355-d079-4fdb-ada0-72b83fae1d2a.JPG)

![excel4](https://user-images.githubusercontent.com/62857660/156849317-c7d62a25-08f3-4d3e-9415-6497b70f6d84.jpg)

- Get random date and time in a range '=RANDBETWEEN(DATE(2020,1,1),DATE(2020,12,31))+RANDBETWEEN(TIME(9,0,0)*10000,TIME(23,0,0)*10000)/10000'
- Get random text `=CHOOSE(RANDBETWEEN(1,3),"Value1","Value2","Value3")`
- Remove duplicates: Data -> next to Text to Column -> click the remove duplicate icon
- Use `filter` to see the different spelling of the same thing, then filter them out to fix it:

![excel](https://user-images.githubusercontent.com/62857660/156689379-525f143b-c94d-413b-99aa-166cdd13b9b9.png)

- `=TRIM()` to trim out white spaces
- Change numbers to another format

![excel2](https://user-images.githubusercontent.com/62857660/156702577-6ecd31b6-f8d2-452c-83c2-fe66e8b76c32.png)

- `=CONCAT("First","","Last")` to get a space between first and last word
- Condtional formatting

![excel3](https://user-images.githubusercontent.com/62857660/156708541-7f80669d-add6-4d2b-b446-1cb6d321330b.jpg)

- Use data validation to only limit values another user can input





## Tableau and Power BI
- Slicer in Power BI works similar to the filter in Tableau

![powerbi](https://user-images.githubusercontent.com/62857660/158744343-c905f3dc-1bc0-4721-9924-45281f0069ad.png)

- Parameter is a workbook variable that can replace a constant value in calculation, filter and reference line. Sort of like a subquery. Use parameter in Top N, moving average, choose date range, 

![Screenshot 2022-03-23 233503](https://user-images.githubusercontent.com/62857660/159844314-2659c566-2cce-48c5-854c-ed03f1cf1e10.png)
![1b](https://user-images.githubusercontent.com/62857660/159844323-5b6e74be-1971-4732-86aa-1c7ebf3ec276.png)![2](https://user-images.githubusercontent.com/62857660/159844336-c450cdf8-603e-4807-b11f-cb36912b8ff7.png)

![4](https://user-images.githubusercontent.com/62857660/159844352-49fa82f7-d2ca-4d4e-b2f3-8cc672cb6315.png)

![5](https://user-images.githubusercontent.com/62857660/159847056-6bb4a78c-fd4e-497a-b032-f8b50eaca6e6.png)

![6](https://user-images.githubusercontent.com/62857660/159847076-ca986bec-7cc3-4a3d-8bdf-12524bb1344f.png)![7](https://user-images.githubusercontent.com/62857660/159847081-d0ab5c96-d78b-4f34-9799-02ac4cdf1b7f.png)

![8](https://user-images.githubusercontent.com/62857660/159847086-2c3876c6-75ba-4c93-828d-9464b600e5f1.png)

![9](https://user-images.githubusercontent.com/62857660/159847093-da64b247-3cca-4cdc-8a65-4fcf759f4859.png)




## Business
How to improve a product? Some steps:
1. Clarify the question: Which feature are we focus on? How the feature actually work?
2. Approach to improve: analyze the user journey map (think of yourself as the user and what would attract or prevent you from the product). Come up with any ideas! Another one is segment users into group
3. Prioritize ideas: quantity analysis, EDA, cost-effective idea
4. Define 1 to 2 success metrics


Moving Average
- Allow to see the pattern in volatile data

KPI
- Key Performance Indicators (KPI) are used to measure a business's performance over a set period of time. The data analyst must decode this information and present it in easy-to-understand terms, allowing organizations to develop more powerful strategies.
- ![1](https://user-images.githubusercontent.com/62857660/160064954-97e21395-2fd4-4255-8d6d-427a0d19e1a4.png)


What Are the 5 Key Performance Indicators?
- Revenue growth.
- Revenue per client.
- Profit margin.
- Client retention rate.
- Customer satisfaction.

More KPI examples
- Customer Acquisition Cost. Customer Lifetime Value. Customer Satisfaction Score. Sales Target % (Actual/Forecast) ...
- Revenue per FTE. Revenue per Customer. Operating Margin. Gross Margin. ...
- ROA (Return on Assets) Current Ratio (Assets/Liabilities) Debt to Equity Ratio. Working Capital.

KPI in SEO
- https://en.ryte.com/magazine/google-analytics-these-are-the-10-most-important-kpis-for-your-website
![internal-kpis](https://user-images.githubusercontent.com/62857660/158462004-fec4c104-cd64-4763-8b7c-706efe68c3c5.jpg)


SCRUM
-  Framework for developing, delivering, and sustaining products in a complex environment.
-  The key difference between Agile and Scrum is that while Agile is a project management philosophy that utilizes a core set of values or principles, Scrum is a specific Agile methodology that is used to facilitate a project
-  Scrum is an Agile project management methodology involving a small team led by a Scrum Master, whose main job is to remove all obstacles to getting work done. Work is done in short cycles called sprints, and the team meets daily to discuss current tasks and any roadblocks that need clearing.

[Back to Top](#machine-learning)
