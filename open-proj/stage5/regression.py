import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn . decomposition import PCA
import time

def randomForestRegressor1(X_train, X_test, y_train, y_test):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    RFR = RandomForestRegressor()
    RFR = RFR.fit(X_train, y_train)
    RFR_train_pred = RFR.predict(X_train)
    RFR_test_pred  = RFR.predict(X_test)
    print('RandomForestRegressor')
    mean_squared_error(y_train, RFR_train_pred),
    mean_squared_error(y_test,  RFR_test_pred)
    print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, DT_train_pred),
    r2_score(y_test,  DT_test_pred)));
    return RFR;


def decisionTreeRegressor1(X_train, X_test, y_train, y_test):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    from sklearn.tree import DecisionTreeRegressor
    DT = DecisionTreeRegressor(max_depth=3)
    DT.fit(X_train,y_train)
    DT_train_pred = DT.predict(X_train)
    DT_test_pred  = DT.predict(X_test)
    print('DecisionTreeRegressor')
    print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, DT_train_pred),
    mean_squared_error(y_test,  DT_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, DT_train_pred),
    r2_score(y_test,  DT_test_pred)));
    return DT;

def ransacregressor(X_train, X_test, y_train, y_test):
    from sklearn . linear_model import LinearRegression
    from sklearn . linear_model import RANSACRegressor
    ransac = RANSACRegressor(LinearRegression() , max_trials=100, min_samples=50 , residual_threshold=5.0, random_state=1)
    ransac.fit(X_train,y_train)
    print('RANSAC Regressor')
    y_train_pred = ransac.predict(X_train)
    y_test_pred = ransac.predict(X_test)
    print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)));
    return ransac

def feature(model,x):
    import matplotlib.pyplot as plt
    importances = model.feature_importances_
    importances_df = pd.DataFrame(
        data = {'column' : x.columns, 'importance' : importances}
    )

    importances_df = importances_df.sort_values(by = 'importance', ascending = False)

    importances_df['weighted'] = importances_df['importance'] / importances_df['importance'].sum()


    plt.figure()
    plt.title("Feature importances")
    (pd.Series(model.feature_importances_, index=x.columns)
       .nlargest(15)
       .plot(kind='barh'))
    plt.show()

def pca(x_train,x_test):
    standard_pca = PCA(n_components=0.8)
    X_train_dim = standard_pca.fit_transform (x_train)
    X_test_dim= standard_pca.transform (x_test)
    return (X_train_dim,X_test_dim)
    
def crosvalscore(model,X,y):
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(estimator=model,
                          X=x_train,
                          y=y_train,
                          cv=10,
                          n_jobs=1)
    print('CV accuracy scores: %s' % scores);
    
def result(model,test_agg):
    import csv
    predictions_test = model.predict(test_agg.drop(['fullVisitorId'], axis = 1))

    submission = pd.DataFrame({
        "fullVisitorId": test_agg['fullVisitorId'].astype(str),
        "PredictedLogRevenue": predictions_test
        })

    submission['fullVisitorId'] = submission['fullVisitorId'].astype(str)
    submission.to_csv('submission_rf.csv', quoting=csv.QUOTE_NONNUMERIC, index = False)

dataset = pd.read_csv('application.csv')
id_train = dataset['fullVisitorId']
x = dataset.drop(['TARGET','fullVisitorId'], axis = 1)
y = dataset['TARGET']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

print("--without dimensionality reduction technique--")
start_time=time.time()
model = randomForestRegressor1(x_train,x_test,y_train,y_test)
end_time = time.time()
print("--- Time taken is %s seconds ---" % (end_time - start_time))
feature(model,x_train)

start_time = time.time()
model1 = decisionTreeRegressor1(x_train,x_test,y_train,y_test)
end_time = time.time()
print("--- Time taken is %s seconds ---" % (end_time - start_time))
feature(model1,x_train)

start_time = time.time()
model2 = ransacregressor(x_train,x_test,y_train,y_test)
end_time = time.time()
print("--- Time taken is %s seconds ---" % (end_time - start_time))

print("\n--with dimensionality reduction technique--")

X_train_dim,X_test_dim = pca(x_train,x_test) 
start_time = time.time()
model = randomForestRegressor1(X_train_dim,X_test_dim,y_train,y_test)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))

start_time = time.time()
model1 = decisionTreeRegressor1(X_train_dim,X_test_dim,y_train,y_test)
print("--- Time taken is %s seconds ---" % (time.time() - start_time));

start_time = time.time()
model2 = ransacregressor1(X_train_dim,X_test_dim,y_train,y_test)
print("--- Time taken is %s seconds ---" % (time.time() - start_time))
     
test_agg = pd.read_csv('test_agg.csv')
result(model,test_agg)

#crosvalscore(model,x,y)

    