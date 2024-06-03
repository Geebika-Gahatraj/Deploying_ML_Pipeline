# Machine Learning Pipeline for StartUps Acquisition
Dataset link: 'https://drive.google.com/file/d/1tWYkHYHm2HoiCajZ49Cs1K7sklWTdAbV/view'

The dataset is all about the Startup’s Financial Information and we need to predict the current financial status of the company.

##### Target Column:
 'status' with classes (operating, closed, acquired,ipo).

 The dataset is highly biased:
 
 operating    93.32%

acquired      4.77%

closed        1.31%

ipo           0.57%

## Understanding the Dataset

##### Features: 
19 columns, including 3 categorical and 16 numerical (including date columns).


##### Data Quality Insights:
Only 9 columns out of 44 has 0 null values.

![Detailed_workflow](https://github.com/Technocolabs100/Building-Machine-Learning-Pipeline-on-Startups-Acquisition/assets/104253294/9f84fa92-6189-4043-be8d-e540cd5755f4)

## Data Preprocessing
Preprocessing is crucial in ensuring that data is accurate, consistent, and appropriate for analysis. It can help to improve the quality of results and make analysis more efficient and effective. We have taken several steps:
#### 1. Handling missing values by dropping all the features that does not provide relavant information and is not necessary for further analysis
```bash
   data.drop(columns=[])
```
Dropped columns : 'id',  'Unnamed: 0.1', 'entity_type', 'entity_id',  'parent_id', 'created_by', 'created_at',  'updated_at' as they are redundant and deleted columns such as 'domain', 'homepage_url', 'twitter_username', 'logo_url', 'logo_width', 'logo_height', 'short_description', 'description', 'overview', 'tag_list', 'name', 'normalized_name', 'permalink', 'invested_companies'.

Dropped column 'ROI' as it contained more than 95% null values.

#### 2. Handling Outliers using IQR (Interquartile Range) for funding_rounds and funding_total_usd
## Data Transformation
*   Convert founded_at, closed_at, first_funded_at, last_funding_at, first_milestone_at ,
       last_milestone_at to years.
```bash
data['column'] = pd.to_datetime(data['column'])
data['column'] = data['column'].dt.year
```
2.   Create new variables


    *   Create new feature isClosed from closed_at and status.
    *   reate new feature 'active_days'

   a. Create new feature isClosed from closed_at and status.


    *   if the value in status is 'operating' or 'ipo', Let's put 1.
    *  Where as if the value is 'acquired' or 'closed', let's put 0.

b. Create new feature 'active_days'

1.   Replacing values:


    *   if the value in status is 'operating' or 'ipo' in closed_at, Let's put 2021.
    *    Where as if the value is 'acquired' or 'closed', let's put 0.

2.   Subtract founded_date from closed_date, and calculate age in days (After calculating active days,
     check contradictory issues we didn't check it before).
3. Then, delete the closed_at column.

#### Handling Null values
1. Imputing numerical values using mean Imputation
```bash
data['column'].fillna(data['column'].mean(), inplace=True)
```
2. Imputing categorical values using mode Imputation
```bash
data['column'].fillna(data['column'].mode().iloc[0],, inplace=True)
```
Finally saved the cleaned dataset in new .csv file.
## EDA











The dataset, named "cleaned_companies.csv," was successfully loaded using the Pandas library, presumably containing pre-processed and cleaned data for our analysis.

Shape of dataset (59987, 19)

```bash
  company.shape
```


We initiated the analysis by obtaining descriptive statistics to summarize the central tendency, dispersion, and shape of the distribution of each numerical feature which include measures such as mean, standard deviation, minimum, maximum, and quartiles for numerical variables, aiding in understanding the basic statistical properties of the dataset.

## Visualization
A powerful tool for exploring and interpreting data, was employed using Matplotlib and Seaborn to create visual representations of the data, such as histograms, box plots, and correlation matrices.

#### Univariate Analysis 
```bash
  sns.histplot(data[columns])
```
It provides a focused examination of individual variables, aiding in identifying patterns, outliers, and trends within the dataset. Plotted histograms were used to observe the distribution of data.

#### Bivariate Analysis 
```bash
   plt.scatterplot(x, y, data = data)
```
A crucial step in exploring relationships between two variables. Our focus was on categorical data, and we utilized bar graphs to visualize the distribution of counts for specific categories.

#### Multivariate Analysis 
##### Correlation matrix
```bash
   sns.heatmap(data.corr(), annot=True)
   ```
Provides a numerical representation of relationships between variables.
##### Scatterplots
```bash
   sns.scatterplot(x=data[col1], y=data[col2], hue=data[col3])
```
##### Pair plots
```bash
   sns.pairplot(data)
```
Illustrates relationships between every pair of features in the dataset.

## Feature Engineering
#### ● Objective
State the main objective or goal of the feature engineering process. For example, improving the
predictive accuracy of a machine learning model, reducing overfitting, or extracting meaningful
insights from the data.
#### ● Execution of Feature Engineering
Data and Target Variable(Status) Separation: The data (X) and the target variable (y) are
separated. The target variable is popped from the DataFrame (cleaned_df)

Label Encoding for Categoricals: Categorical features in the dataset are label-encoded using the
factorize method.
 #### ● Mutual Information (MI Scores)
 It was used to find the which feature has a strong relation with the target variable. This is useful for finding the features that can help to build a more interpretable and potentially more predictive model.

 #### ● Principal Component Analysis (PCA)
  Although it comes under the Unsupervised Machine Learning category.
  PCA provides a complete explanation of the composition of variance and covariance using multiple linear combinations of the core variables.
  According to the MI scores we selected the top five features and scaled them using 'MinMax Scaler'. This step is important for PCA because PCA is sensitive to the scale of the features.
  PCA is applied to the scaled features to reduce the dimensionality of the data while preserving most of its variance. PCA does this by transforming the original features into a new set of uncorrelated variables called principal components.The transformed data from PCA is converted back to a DataFrame where each row corresponds to an observation from the original dataset, but with its features transformed into the principal component space.
  
## Model Building
### 1.Binary classification
**Linear Regression**
```bash
   LinearRegression()
```
Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.It is used for solving the classification problems.Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.

#### ●Train-Test-Split
```bash
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=42)
```
#### ● Accuracy score on training data : 97.56%

#### ●Accuracy score on testing data : 97.47%
#### ●Confusion Matrix
True positive : 10877

True negative : 653
#### ● Cross Validation using stratifiedKFold
```bash
   stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```
Mean Accuracy:  97.53%

Standard Deviation of Accuracy:  0.0022 (depicts the model generalizes well on unseen data)
#### ● Hyperparameter Tuning
1. GridSearchCV
2. RandomizedSearchCV 

### 2. Multiclass classification
**Gradient Boosting**
```bash
   GradientBoostingClassifier()
```
Gradient boosting is a machine learning ensemble technique that combines the predictions of multiple weak learners, typically decision trees, sequentially. It aims to improve overall predictive performance by optimizing the model’s weights based on the errors of previous iterations, gradually reducing prediction errors and enhancing the model’s accuracy.

#### ●Accuracy score : 97.68%

#### ● Hyperparameter Tuning
The accuracy canbe improved by tuning hyperparameters or processing data to remove outliers.

**Best parameters :
learning_rate: 0.1, n_estimators: 50,
max_depth: 4**



## Pipeline
The pipeline integrates various preprocessing steps and machine learning algorithms to predict whether a company is open or closed based on its features.
### 1. Pipeline components
#### 	Standard Scaler:
```bash
   stdscaler = StandardScaler()
``` 
Standardised the features by removing the mean                          and scaling to unit variance.
#### 	Principal Component Analysis (PCA):
```bash
   pca = PCA(n_components = 9)
``` 
Reduces the dimensionality of the feature space while preserving most of the variance in the data.
#### 	Classifier (Logistic Regression/Gradient Boosting):
```bash
   Log_regression = LogisticRegression()
   Grad_boosting = GradientBoostingClassifier()
```
Implements the classification algorithm to predict company status. 
#### 	RandomOverSampler (for Logistic Regression):
```bash
   over = RandomOverSampler(random_state=0)

```
Resamples the training data to balance the classes by randomly replicating minority class samples.
#### 	SMOTE (for Gradient Boosting and Ensemble Learning):
```bash
smote = SMOTE()
```
Generates synthetic samples of the minority class to balance the class distribution.
### 2. Pipeline Execution
#### 	Logistic Regression
```bash
pipe = Pipeline([
    ('scaler', stdscaler),
    ('pca', pca),
    ('classifier', LogisticRegression(C= 1.623776739188721, penalty= 'l2', solver= 'newton-cg'))
])
```
 Test accuracy = 72.4%.

#### 	Gradient Boosting
```bash
pipeline_grad_boosting = Pipeline([
    ('scaler', stdscaler),
    ('pca', pca),
    ('classifier', GradientBoostingClassifier(learning_rate=0.1,n_estimators=60,max_depth=3))
])
```

Test accuracy = 65.63%.
#### 	Ensemble Learning
```bash
model_ensemble = Pipeline([
    ('stdscaler', stdscaler),
    ('pca',pca),    
    ('classifier', VotingClassifier([('Logistic Regression',Log_regression), ('Gradient Boosting', Grad_boosting)]))
])
```
 Test accuracy = 79.58%.
### 3. Performance Comparison
●	Logistic regression achieves moderate accuracy but is outperformed by Ensemble Learning.

●	Gradient Boosting shows slightly lower accuracy compared to Logistic Regression.

●	Ensemble Learning combines the strengths of Gradient Boosting, resulting in improved accuracy.

## Deployment using Flask and Render

#### Flask
Flask is a small and lightweight Python web framework that provides useful tools and features that make creating web applications in Python easier.
We write a python code for our app using flask; the app asks the user to enter the data of the company. We only used 9 datas from the user as input.
Ensemble model was used for prediction of lable from inputdata and class mapping was done.
The output will be **Operating**, **Acquired**, **Closed** and **IPO**.

#### Render 
Finally we deployed on the flask server on the internet using Render. You can access our webpage using this link
[https://deploying-ml-pipeline-1.onrender.com]
