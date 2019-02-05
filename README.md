# Credit-Card-Fraud-Detection
Python code to detect Credit Card Fraud.

## Overview
The dataset is taken from a kaggle competition with the same name.<br/>
MinMaxScaler and PCA used for feature scaling and selection respectively.<br/>
Logistic Regression is used for classification.<br/>
Precision, Recall and F1 Score are used as classification metrics.

## Data Description
The dataset can be found here https://www.kaggle.com/mlg-ulb/creditcardfraud.<br/>
The dataset includes 31 columns each having 284807 entries.<br/>
The first column is a **Time** column which doesn't seem to be contributing much to the dataset.<br/>
The columns after time have key ranging from **V1 to V28**. They are not described further due to security concerns.<br/>
After the *V28* column, we are given an **Amount** column which tells us the transaction amount for that particular case.<br/>
Finally, we have **Class** column which if *1* means that the transaction is a fraud transaction and *0* if otherwise.

## Results
> **Recall**    0.7222222222222222<br/>
> **Precision** 0.9154929577464789<br/>
> **F1 Score**  0.8074534161490683

## Contributing
Your contributions are always welcome.<br/>
Feel free to improve existing code, documentation or implement new algorithm.<br/>
Metrics keep changing as soon as the code is run again. Any insights on that?

###### Thanks for reading.
