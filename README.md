# ML_Credit_Card_Fraud
Predicting Fraudulent Transactions with Machine Learning


# Machine Learning with Imbalanced Data - Credit Card Fraud

In the world today, with the modernization and advancements in technology, the volume of credit and debit card transactions has increased significantly having surpassed cash payments. 
Research conducted by the Federal Reserve in 2017, found that global card payments reached 111.1 billion transactions with a value of $5.98 trillion in 2016. Of this, credit card growth rate was at the highest at 10.2 percent. 
Another study by SHIFT credit card processing, shared that $24.26 billion was lost globally due to payment fraud in 2018, showing an increase by 18.4 percent. These statistics highlight a growing problem not only for companies liable for these transactions, 
but to customers harmed by identity theft and fraud. The aim of our research is to utilize predictive models within Python and evaluate their performance with regards to solving this growing business problem.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Jupyter Notebook
Libraries: IMBlearn, SciKitLearn, Pandas, Seaborn, Matplotlib
```
## Data 

The data utilized was sourced from Kaggle, which came about for a research collaboration between Worldline Payment Services and Université Libre de Bruxelles for machine learning purposes. 
The dataset holds 284,807 transactions made by European credit cardholders during the month of September 2013, of which 492 or 0.173% of all transactions are fraud. 
The 30 features include Time (Seconds elapsed between each transaction and the first transaction), Amount (Transaction amount), and V1 – V28, which were derived from Principal Component Analysis (PCA) for dimensionality reduction and normalized. 
Due to privacy concerns for cardholders, further details regarding the origins of V1-V28 could not be shared. Finally, the data contains a target label named Class, which is a binary variable for whether the transaction was Fraudulent (1) or Non-Fraudulent (0). 

## Exploratory Data Analysis

The first steps of our analysis included descriptive analysis and visualization to better understand the data and features. After examining each feature for missing values, we found there to be no missing values which helped simplify the preprocessing.  A count plot for the target variable helps to visualize the class imbalance mentioned previously, which informs our next decision of how to approach handling the data. Class imbalance is a challenge faced by practitioners in machine learning, where if the model is trained on imbalanced data, it skews and biases the model and subsequent results. There are many techniques to combat this challenge, including under-sampling, over-sampling, synthetic data, and utilizing certain machine learning models. In our analysis, we examined under-sampling, synthetic over-sampling (SMOTE), and different machine learning algorithms, which all come with their own advantages and disadvantages.  SMOTE is a statistical technique for increasing the number of fraud cases in our data set in a balanced way. SMOTE used the original fraud cases to synthetically replicate new fraud cases in our training data set. Under sampling removes records from the data to reduce the class imbalance from the data set. In our analysis, we tested the models on the data provided from both under sampling and SMOTE. For the scope of the project, we reduced the number of observations to 71,578 while retaining all fraud data to speed the training of our models.

![Image of Distrib](https://picturesadblog.s3-us-west-2.amazonaws.com/dist_imb.png)
Visual of distribution among features

![Image of Outliers](https://picturesadblog.s3-us-west-2.amazonaws.com/imb_dist.png)
Visual of Outliers among features

## Preprocessing

While most of our data has already gone through extensive preprocessing in the form of standardization and PCA, there were still some preprocessing required prior to model building. Firstly, as Time and Amount were left untouched, we needed to scale the data to a similar level as the rest of the data. From our EDA, we found that the Amount feature had multiple outliers within fraudulent and non-fraudulent data, which is why we chose to use the robust scaler to reduce their influence. Following this, we examined potential anomaly and outlier detection techniques with the features V1 – V28. The two considered were the Standard Deviation Method for Gaussian distributions and Inter-Quartile Range (IQR) Method for Non-Gaussian Distributions. Given the distributions, we removed outliers with the IQR method. 

## Predictive Models

The goal of the research was to develop and test various predictive models to evaluate their performance in solving the business problem described. It was important that the models be able to maximize both precision and recall, as the ability to detect fraud correctly is just as important as not incorrectly detecting fraud (False Negative). In the business world this makes sense, as we want to detect fraud to save the business money, however we don’t want to incorrectly detect fraud consistently as we would annoy and lose future customers. There is a trade-off between these two measures, and we will further describe our performance evaluation measures at a later stage to best address this. As the problem is a classification problem, which aims to predict a binary output of Fraud (1) and Non-Fraudulent (0), we will use classification algorithms. The five models utilized include K Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), Decision Trees, and Random Forest.  
## Authors

* **Adrian Leung** - *Initial work* - (https://github.com/leungad)

