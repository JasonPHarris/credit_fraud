# credit_fraud
Using creditfraud.arff I have written a machine learning algorithm that predicts credit card fraud. 

I imported the arff file using scipy.io and then loaded it into pandas. 
Then I used the tutorial 
(https://dataaspirant.com/credit-card-fraud-detection-classification-algorithms-python/#:~:text=The%20decision%20tree%20is%20the,up%20with%20the%20important%20features.) 
that I found online to establish a good basis for my code and algorithms, 
but I also came up with completely different results than the tutorial, due to using a different data set than the tutorial. 
I also had to use sklearn LabelEncoder to transform object data types into int types, and StandardScaler to limit the range of data on 
the credit_usage and current_balance columns so that they do not skew the results. I also used the file converter that I 
located at https://pulipulichen.github.io/jieba-js/weka/arff2csv/ to convert the arff to a csv to view the contents of the file in 
Excel while working with the data.

Using the Random Forest and Decision Tree Classifiers, I have implemented the algorithms in Python to ingest, prepare, model, and show the 
results of the algorithms. I chose Random Forest because it has been stated by a few people online that it is one of the better supervised 
algorithms for fraud detection. I chose Decision Tree Classifiers as well, because of my personal curiosity to see how the two compare in 
this scenario. I have used a training, testing, and validation approach that allows for the algorithm to be run on future unseen data. 
Training and testing are done in the standard fashion using sklearn.model_selection train_test_split on the raw data initially. 
After comparing the algorithms, I used under sampling of the data to run the two algorithms again but used the roc_auc_score in addition 
to the Classification report to compare the accuracy of the models. Unfortunately, both of my models performed below 80% accuracy. 
Decision Trees performed at 72.5% while Random Forests performed at 76.2%.
