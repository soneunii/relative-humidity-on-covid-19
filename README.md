# Relative Humidity On COVID-19 
In order to have information on the weather on the particular day a Covid19 infection case
occurred, I merged the two datasets by using the date when symptoms started to show of each case as a
key to find the corresponding weather. From this combined dataset, I use (1) confirmed date, (2) symptom
onset date, (3) province, (4) average relative humidity, (5) average temperature, (6) minimum
temperature, (7) maximum temperature, (8) precipitation, (9) maximum wind speed, and (10) most
frequent wind direction. 87% of cases were missing symptom onset dates (2), so I subtracted 7 days,
which is in average the time it takes for symptoms to develop, from the confirmed date (2) to fill the
missing dates that I will use. Also, I eliminated the 3 rows where the dataset was missing confirmed dates
(1).

## Number of Cases and Relative Humidity Rates 
After creating a dataset that I can work on, I compared the number of cases per province in which
the average relative humidity (4) was between 40% and 60%, as seen on Figure 1.  I created a separate
dataset where I group the data by province, and re-calculate the average relative humidity for each
province, instead of each day, and only including the cases that occurred when the average relative
humidity was between the range. In this figure, it is prominent to see that regions with relatively high
temperatures have more days that has a relative humidity between the 40% to 60% range, leaving them
with a wider timeframe to have more cases occur, when looking at the total height of each bar.


## Number of Components vs. Cumulative Explained Variance 
In order to check the data’s features, I used a principal component analysis over features 5-10. As
seen on Figure 2, the cumulative explained variance ratio of both ways, scaled and not scaled, as more
features are introduced into the pipelines. The first feature after being scaled, yields 51% variance, which
is relatively higher than the other 5 features, scaled. This makes sense because if the features were
compared unscaled, it would be comparing the average temperature (5), which is measured in Celsius,
and the most frequent wind direction (10), which is measured in degrees, as in angles. The columns show
that the features all together contain 98% of the information. This shows that data should be used for
regression, as it shows that there are limitations for it to not lose significant information while reducing
the dimensions.

## Logistic Regression Coefficients 
Lastly, Figure 3 shows the variables that were used for the model that predicted the average
relative humidity (4), and the coefficients that were used for regression. I created two models to determine
on which would perform better, one of which uses just a single linear regression, while the other has a
polynomial feature added before regression and fitting. Comparing the two model’s scores, I determined
to use the model using only the linear regression, as it scored approximately 9% better. This data shows
that the maximum temperature (7) of the day has the greatest negative weight on determining the average
relative humidity (4), yielding -1.58. At the same time, the minimum temperature (6) shows to have the
greatest positive weight by 1.23.

Based on the findings in this project, relative humidity levels are only one component that can
determine whether or not the virus is more likely to spread or not. Under various uncertainties during this
pandemic, it is important for us to keep looking for relationships between weather and the spread of a
virus to identify specific conditions the virus is more likely to stay. 
