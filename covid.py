import pandas as pd
import requests
import os
import numpy as np
from collections import defaultdict
from datetime import timedelta
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import make_column_transformer

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import datetime
from datetime import date, timedelta

patient_df = pd.read_csv("patient_info.csv")
patient_df=patient_df.drop(['deceased_date'], axis=1)
patient_df = patient_df[patient_df['confirmed_date'].notna()]
weather_df = pd.read_csv("weather.csv")

#updating the 'None' values in 'symptom_onset_date' column to 7 days beore 'confirmed date'
patient_df['confirmed_date'] = pd.to_datetime(patient_df['confirmed_date'],format='%Y-%m-%d')

for date in patient_df['confirmed_date']:
date=date.date()
patient_df['symptom_onset_date'] = patient_df['symptom_onset_date'].fillna(patient_df['confirmed_date'] - pd.to_timedelta(7, unit='d'))

for i, date in enumerate(patient_df['symptom_onset_date']):
if isinstance(date, datetime.datetime) == True:
date= str(date)
patient_df.iloc[i]['symptom_onset_date'] = date[:10]

df = patient_df.merge(weather_df,left_on=['province','symptom_onset_date'],right_on=['province','date'])
df = df.drop(['code','released_date','date','confirmed_date','infected_by'],axis=1)


#creating data frame that counts the number of cases per province in which avg_humidity was between 40% and 60% 
count = 0
ncount=0
ncount_dict={}
count_dict = {}
for i in range(len(df)):
  if df.iloc[i]['avg_relative_humidity']>= 40 and df.iloc[i]['avg_relative_humidity']<=60:
    count+=1
  count_dict[df.iloc[i]['province']]= count
for i in range(len(df)):
  if df.iloc[i]['avg_relative_humidity']< 40 or df.iloc[i]['avg_relative_humidity']>60:
    ncount+=1
  ncount_dict[df.iloc[i]['province']]= ncount
ncount_df = pd.Series(ncount_dict).to_frame()
ncount_df.index.names = ['province']
ncount_df.columns=['case_out_of_range']
count_df = pd.Series(count_dict).to_frame()
count_df.index.names = ['province']
count_df.columns=['case_within_range']
case_merged = pd.merge(count_df,ncount_df,on='province')

#finding average relative humidity 
sum_humidity = pd.DataFrame(df.groupby(['province']).sum()['avg_relative_humidity'])
occur = pd.DataFrame(df.province.value_counts())['province']
occur=pd.DataFrame(occur)
occur.rename(columns = {'province':'reoccur_province'}, inplace = True)
occur.index.names = ['province']
merged_df=pd.merge(occur,sum_humidity,on='province')
merged_df['avg'] = (merged_df['avg_relative_humidity']/merged_df['reoccur_province']).round(2)
merged_df=pd.merge(merged_df,case_merged,on='province')
merged_df.sort_values(by=['case_within_range','case_out_of_range'],ascending=[False,False])

#plot1
x =['Gyeonggi-do','Seoul','Busan','Chungcheongnam-do','Gyeongsangnam-do','Daejeon','Gyeongsangbuk-do','Daegu','Incheon','Gwangju','Ulsan','Jeollabuk-do']
y1 = merged_df['case_within_range']
y2 = merged_df['case_out_of_range']
plt.bar(x, y1, color='g')
plt.bar(x, y2, bottom=y1, color='y')
plt.xticks(rotation=90)
plt.xlabel('Province')
plt.ylabel('Number of Cases')
plt.title('Figure 1: Number of Cases and Relative Humidity Rate')
plt.legend(['case_within_range','case_out_of_range'])
xcols =["avg_temp","min_temp","max_temp","precipitation","max_wind_speed","most_wind_direction"]
ycol = "avg_relative_humidity"

#creating line plot where x-axis is the number of components and y axis is the cumulative explained variance (PCA with any number of dimensions) 
def explained(scale):
  stages = [("imp", SimpleImputer(strategy="most_frequent"))]
  if scale:
    stages.append(("std", StandardScaler()))
  stages.append(("pca", PCA()))
  p = Pipeline(stages)
  p.fit(df[xcols])
  explained = p["pca"].explained_variance_
  print(p["pca"].explained_variance_ratio_)
  s = pd.Series(explained.cumsum() / explained.sum(),index=range(1, len(xcols)+1))
  return s

#plot2
ax = explained(False).plot.line(label="Not Scaled", ylim=0,color='g')
explained(True).plot.line(label="Scaled",color='y', ax=ax)
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, loc = 'lower right')
ax.set_title("Figure 2: Principal Components", pad=20)


xcols =["avg_temp","min_temp","max_temp","precipitation","max_wind_speed","most_wind_direction"]
ycol = "avg_relative_humidity"
train_df, test_df = train_test_split(df,random_state=0)

#model 1: linear regression 
model_1 = Pipeline([
('lr',LinearRegression())
])

model_1.fit(train_df[xcols],train_df[ycol])
scores1 = model_1.score(test_df[xcols],test_df[ycol])
scores1.mean()

#model 2:Polynomial Features Linear Regression 
model_2 = Pipeline([
('poly',PolynomialFeatures(degree=2,include_bias=False)),
('lr',LinearRegression())
])

model_2.fit(train_df[xcols],train_df[ycol])
scores2 = model_2.score(test_df[xcols],test_df[ycol])
scores2.mean() 

#plot 3
idx = [t.replace("_", " ") for t in xcols]
ax = pd.Series(model_1["lr"].coef_, index=idx).plot.barh(color='g')
ax.set_xlabel("Weight")
ax.set_ylabel("Feature")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Figure 3: Logistic Regression Coefficients", pad=20)
