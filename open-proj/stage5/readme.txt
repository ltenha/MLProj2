
Google Analytics Customer Revenue Prediction
Predict how much GStore customers will spend

Codes to run the files

to visualize the attributes from the dataset
-python trainanalysis.py

to preprocess the dataset, create a new dataset named 'application.csv' which has the aggregated values for each customer and the test_agg file for prediction
-python preprocessing.py

to apply the regression models, see the scores and save the predicted revenue value with visitorId in a csv file
-python main.py

Dataset files:
train.csv ('http://test.blueathiean.me/train.csv')
test.csv ('http://test.blueathiean.me/test.csv')

Intermediate csv files
application.csv
train_agg.csv

final csv file with visitorId and predicted revenue
submission_rf.csv
