import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime as datetime
import seaborn as sns

def trainanalysis(train_data):
    train_data.transactionRevenue = train_data.transactionRevenue.astype(float)
    target= np.log(train_data.transactionRevenue.dropna())

    revenue_datetime_df = train_data[["transactionRevenue" , "date"]]
#revenue_datetime_df["transactionRevenue"] = revenue_datetime_df.transactionRevenue.astype(np.int64)

    total_revenue_daily_df = revenue_datetime_df.groupby(by=["date"],axis=0).sum()

    total_visitNumber_daily_df = train_data[["date","visitNumber"]].groupby(by=["date"],axis=0).sum()

    datetime_revenue_visits_df = pd.concat([total_revenue_daily_df,total_visitNumber_daily_df],axis=1)

#plotting graphs for different features of dataset

    fig, axes = plt.subplots(1,1,figsize=(10,5))
    train_data.channelGrouping.value_counts().plot(kind="bar",title="channelGrouping distro",rot=25,colormap='Paired')

    fig, axes = plt.subplots(2,2,figsize=(15,15))
    train_data["isMobile"].value_counts().plot(kind="bar",ax=axes[0][0],rot=25,title="isMobile",color='tan')
    train_data["browser"].value_counts().head(10).plot(kind="bar",ax=axes[0][1],rot=40,title="browser",color='teal')
    train_data["deviceCategory"].value_counts().head(10).plot(kind="bar",ax=axes[1][0],rot=25,title="deviceCategory",color='lime')
    train_data["operatingSystem"].value_counts().head(10).plot(kind="bar",ax=axes[1][1],rot=80,title="operatingSystem",color='c')

    fig, axes = plt.subplots(3,2, figsize=(15,15))
    train_data["continent"].value_counts().plot(kind="bar",ax=axes[0][0],title="Global Distributions",rot=0,color="c")
    train_data[train_data["continent"] == "Americas"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][0], title="America Distro",rot=0,color="tan")
    train_data[train_data["continent"] == "Asia"]["subContinent"].value_counts().plot(kind="bar",ax=axes[0][1], title="Asia Distro",rot=0,color="r")
    train_data[train_data["continent"] == "Europe"]["subContinent"].value_counts().plot(kind="bar",ax=axes[1][1],  title="Europe Distro",rot=0,color="lime")
    train_data[train_data["continent"] == "Oceania"]["subContinent"].value_counts().plot(kind="bar",ax = axes[2][0], title="Oceania Distro",rot=0,color="teal")
    train_data[train_data["continent"] == "Africa"]["subContinent"].value_counts().plot(kind="bar" , ax=axes[2][1], title="Africa Distro",rot=0,color="silver")

    fig, axes = plt.subplots(1,1,figsize=(20,10))
    axes.set_ylabel("# of visits")
    axes.set_xlabel("date")
    axes.set_title("Daily Visits")
    axes.plot(total_visitNumber_daily_df["visitNumber"])


    fig, ax1 = plt.subplots(1,1, figsize=(20,10))
    t = datetime_revenue_visits_df.index
    s1 = datetime_revenue_visits_df["visitNumber"]
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('day')
    ax1.set_ylabel('visitNumber', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    s2 = datetime_revenue_visits_df["transactionRevenue"]
    ax2.plot(t, s2, 'r--')
    ax2.set_ylabel('revenue', color='r')
    ax2.tick_params('y', colors='r')
#fig.tight_layout()

    plt.figure(figsize=(12,7))
    sns.countplot(train_data.source,order=train_data.source.value_counts().iloc[:50].index)
    plt.xticks(rotation=90);


    plt.show()

def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)

    for column in JSON_COLUMNS:
        df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

    return df

def preprocessing(train_data):
    train_data["date"] = pd.to_datetime(train_data["date"],format="%Y%m%d")
    train_data["visitStartTime"] = pd.to_datetime(train_data["visitStartTime"],unit='s')
    train_data=train_data.drop(['socialEngagementType'],axis=1)
    
    return train_data
    
train_data=load_df('http://test.blueathiean.me/train.csv')
train_data = preprocessing(train_data)
trainanalysis(train_data) 
