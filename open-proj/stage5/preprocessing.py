import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import timedelta, date
import seaborn as sns
import matplotlib.cm as CM
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)

    for column in JSON_COLUMNS:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

    return df

def preprocessing(train_data,test_data):
    test_data["date"] = pd.to_datetime(test_data["date"],format="%Y%m%d")
    test_data["visitStartTime"] = pd.to_datetime(test_data["visitStartTime"],unit='s')
    test_data=train_data.drop(['socialEngagementType'],axis=1)

    train_data["date"] = pd.to_datetime(train_data["date"],format="%Y%m%d")
    train_data["visitStartTime"] = pd.to_datetime(train_data["visitStartTime"],unit='s')
    train_data=train_data.drop(['socialEngagementType'],axis=1)

    return train_data,test_data

def encoder(train_data):
    from sklearn import preprocessing
    networkDomain_encoder =  preprocessing.LabelEncoder()
    keyword_encoder =  preprocessing.LabelEncoder()
    referralPath_encoder =  preprocessing.LabelEncoder()
    city_encoder =  preprocessing.LabelEncoder()
    visitNumber_encoder =  preprocessing.LabelEncoder()
    source_encoder =  preprocessing.LabelEncoder()
    region_encoder =  preprocessing.LabelEncoder()
    date_encoder =  preprocessing.LabelEncoder()
    country_encoder =  preprocessing.LabelEncoder()
    metro_encoder =  preprocessing.LabelEncoder()
    browser_encoder =  preprocessing.LabelEncoder()
    adContent_encoder =  preprocessing.LabelEncoder()
    subContinent_encoder =  preprocessing.LabelEncoder()
    operatingSystem_encoder =  preprocessing.LabelEncoder()
    campaign_encoder =  preprocessing.LabelEncoder()
    networkDomain_encoder.fit(train_data['networkDomain'])

    train_data['keyword'].fillna(value = '0', inplace = True)
    keyword_encoder.fit(train_data['keyword'])
    train_data['referralPath'].fillna(value = '0', inplace = True)
    referralPath_encoder.fit(train_data['referralPath'])
    city_encoder.fit(train_data['city'])
    visitNumber_encoder.fit(train_data['visitNumber'])
    source_encoder.fit(train_data['source'])
    region_encoder.fit(train_data['region'])
    date_encoder.fit(train_data['date'])
    country_encoder.fit(train_data['country'])
    metro_encoder.fit(train_data['metro'])
    browser_encoder.fit(train_data['browser'])

    train_data['adContent'].fillna(value = '0', inplace = True)
    adContent_encoder.fit(train_data['adContent'])
    subContinent_encoder.fit(train_data['subContinent'])
    operatingSystem_encoder.fit(train_data['operatingSystem'])
    campaign_encoder.fit(train_data['campaign'])

    train_data['networkDomain_encoder'] = networkDomain_encoder.transform(train_data['networkDomain'])
    train_data['keyword_encoder'] = keyword_encoder.transform(train_data['keyword'])
    train_data['referralPath_encoder'] = referralPath_encoder.transform(train_data['referralPath'])
    train_data['city_encoder'] = city_encoder.transform(train_data['city'])
    train_data['visitNumber_encoder'] = visitNumber_encoder.transform(train_data['visitNumber'])
    train_data['source_encoder'] = source_encoder.transform(train_data['source'])
    train_data['region_encoder'] = region_encoder.transform(train_data['region'])
    train_data['date_encoder'] = date_encoder.transform(train_data['date'])
    train_data['country_encoder'] = country_encoder.transform(train_data['country'])
    train_data['metro_encoder'] = metro_encoder.transform(train_data['metro'])
    train_data['browser_encoder'] = browser_encoder.transform(train_data['browser'])
    train_data['adContent_encoder'] = adContent_encoder.transform(train_data['adContent'])
    train_data['subContinent_encoder'] = subContinent_encoder.transform(train_data['subContinent'])
    train_data['operatingSystem_encoder'] = operatingSystem_encoder.transform(train_data['operatingSystem'])
    train_data['campaign_encoder'] = campaign_encoder.transform(train_data['campaign'])

    return train_data

def valuetoint(train_data):
    train_data['visitId'].astype(str).astype(int)
    train_data['visitNumber'].astype(str).astype(int)
    train_data['hits'] = train_data['hits'].astype(int)
    train_data['pageviews'].fillna(value = '0', inplace = True)
    train_data['pageviews'] = train_data['pageviews'].astype(int)
    return train_data

def one_hot_encode(train_data,test_data):
    train_one_hot = train_data[
    [
        'channelGrouping',
        'deviceCategory',
        'isMobile',
        'language',
        'continent',
        'medium',
        'newVisits',
        'visits',
        'campaignCode',
        'isTrueDirect',
        'bounces'
    ]]

    train_one_hot = pd.get_dummies(train_one_hot)

    train_data = pd.concat(
    [
        train_data,
        train_one_hot
    ],
    axis = 1)


    test_one_hot = test_data[
    [
        'channelGrouping',
        'deviceCategory',
        'isMobile',
        'language',
        'continent',
        'medium',
        'newVisits',
        'visits',
        'isTrueDirect',
        'bounces'
    ]]

    test_one_hot = pd.get_dummies(test_one_hot)

    test_data = pd.concat(
    [
        test_data,
        test_one_hot
    ],
    axis = 1)

    del train_one_hot
    del test_one_hot
    return train_data,test_data

def datetimeconvert(train_data):
    train_data['date'] = pd.to_datetime(train_data['date'], format = '%Y%m%d')
    train_data['month'] = pd.DatetimeIndex(train_data['date']).month
    train_data['year'] = pd.DatetimeIndex(train_data['date']).year
    train_data['day'] = pd.DatetimeIndex(train_data['date']).day
    train_data['quarter'] = pd.DatetimeIndex(train_data['date']).quarter
    train_data['weekday'] = pd.DatetimeIndex(train_data['date']).weekday
    train_data['weekofyear'] = pd.DatetimeIndex(train_data['date']).weekofyear
    train_data['is_month_start'] = pd.DatetimeIndex(train_data['date']).is_month_start
    train_data['is_month_end'] = pd.DatetimeIndex(train_data['date']).is_month_end
    train_data['is_quarter_start'] = pd.DatetimeIndex(train_data['date']).is_quarter_start
    train_data['is_quarter_end'] = pd.DatetimeIndex(train_data['date']).is_quarter_end
    train_data['is_year_start'] = pd.DatetimeIndex(train_data['date']).is_year_start
    train_data['is_year_end'] = pd.DatetimeIndex(train_data['date']).is_year_end

    train_data['visitStartTime'] = pd.to_datetime(train_data['visitStartTime'], unit = 's')
    train_data['hour'] = pd.DatetimeIndex(train_data['visitStartTime']).hour
    train_data['minute'] = pd.DatetimeIndex(train_data['visitStartTime']).minute
    return train_data

def delcols(train_data,test_data):
    train_staging = train_data.select_dtypes(exclude = ['object'])
    train_staging = train_staging.select_dtypes(exclude = ['datetime'])
    train_staging = train_staging.select_dtypes(exclude = ['bool'])

    test_staging = test_data.select_dtypes(exclude = ['object'])
    test_staging = test_staging.select_dtypes(exclude = ['datetime'])
    test_staging = test_staging.select_dtypes(exclude = ['bool'])

    return(train_staging,test_staging)

def fillnans(train_staging,test_staging):
    from sklearn.preprocessing import Imputer
    train_staging_columns = train_staging.columns
    imputer = Imputer(strategy = 'mean')
    train_staging = imputer.fit_transform(train_staging)
    train_staging = pd.DataFrame(
        data = train_staging,
        columns = train_staging_columns
    )

    test_staging_columns = test_staging.columns

    test_staging = imputer.fit_transform(test_staging)
    test_staging = pd.DataFrame(
        data = test_staging,
        columns = test_staging_columns
    )
    return(train_staging,test_staging)

def create_target(rev):
    if rev == 0:
        return 0
    else:
        return math.log(rev)

def allinone():
    train_data=valuetoint(load_df('http://test.blueathiean.me/train.csv'))
    test_data = valuetoint(load_df('http://test.blueathiean.me/test.csv'))

    train_data['transactionRevenue'].fillna(value = '0', inplace = True)
    train_data['transactionRevenue'] = train_data['transactionRevenue'].astype(int)

    train_data,test_data = preprocessing(train_data,test_data)

    train_data = encoder(train_data)
    test_data = encoder(test_data)

    train_data,test_data = one_hot_encode(train_data,test_data)

    train_data = datetimeconvert(train_data)
    test_data = datetimeconvert(test_data)
    train_staging,test_staging = delcols(train_data,test_data)

    train_staging,test_staging = fillnans(train_staging,test_staging)

    train_staging, test_staging = train_staging.align(test_staging, join = 'inner', axis = 1)
    train_staging['transactionRevenue'] = train_data['transactionRevenue']
    test_staging['fullVisitorId'] = test_data['fullVisitorId']
    train_staging['fullVisitorId'] = train_data['fullVisitorId']

    train_agg = train_staging \
        .groupby(['fullVisitorId']) \
        .agg(['count','mean','min','max','sum']) \
        .reset_index()

    test_agg = test_staging \
        .groupby(['fullVisitorId']) \
        .agg(['count','mean','min','max','sum']) \
        .reset_index()

    columns_train = ['fullVisitorId']
    for var in train_agg.columns.levels[0]:
        if var != 'fullVisitorId':
            for stat in train_agg.columns.levels[1][:-1]:
                columns_train.append('%s_%s' % (var, stat))

    train_agg.columns = columns_train

    columns_test = ['fullVisitorId']

    for var in test_agg.columns.levels[0]:
        if var != 'fullVisitorId':
            for stat in test_agg.columns.levels[1][:-1]:
                columns_test.append('%s_%s' % (var, stat))

    test_agg.columns = columns_test

    del train_staging
    del train_data

    del test_staging
    del test_data

    train_agg['TARGET'] = train_agg['transactionRevenue_sum'].apply(create_target)

    train_agg = train_agg.drop(
        [
            'transactionRevenue_count',
            'transactionRevenue_mean',
            'transactionRevenue_min',
            'transactionRevenue_max',
            'transactionRevenue_sum'
        ],
        axis = 1
    )
    train_agg_corr = train_agg.corr()

    #CORRELATION CHECK
#Now that we've gotten the data all cleaned up, let's see what kind of signal is in this data set out of the box!
#You will see pageviews / hits are strongest correlations to the revenue the customer spends.
#Intuitively this makes sense: if they click around the site more, it's more likely it will end up in a transaction
    print(train_agg_corr['TARGET'].sort_values(ascending = False))
    train_agg.to_csv('application.csv')
    test_agg.to_csv('test_agg.csv')

allinone()
