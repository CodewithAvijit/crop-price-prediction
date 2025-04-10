import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from datetime import datetime
def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].dropna()


data=pd.read_csv("rawdata\CROP.csv")

oe=OneHotEncoder()
arr=oe.fit_transform(data[["Grade"]]).toarray()
newdata=pd.DataFrame(arr,columns=oe.get_feature_names_out(["Grade"]))
data=pd.concat([newdata,data],axis=1)
data=data.drop(columns=['Grade']) 
data=data.drop(columns=['Arrival_Date','Market'])

# data12=pd.read_csv("rawdata\CROP.csv")
# data12=data12[['Modal_x0020_Price']]
# data=pd.concat([data,data12],axis=1)
# data.to_csv("processdata\CLEAN_CROP.csv",index=False)
data=remove_outliers(data,'Min_x0020_Price')
data=remove_outliers(data,'Max_x0020_Price')
#plt.boxplot(x=data[['Min_x0020_Price','Max_x0020_Price','Modal_x0020_Price']])
#sns.pairplot(data[['Min_x0020_Price', 'Max_x0020_Price', 'Modal_x0020_Price']])
#plt.show()
# data['AVG_Price']=(data['Min_x0020_Price']+data['Max_x0020_Price'])/2
print(data.info())
data.to_csv("processdata\CLEAN_CROP.csv",index=False)
plt.scatter(data['Min_x0020_Price'],data['Modal_x0020_Price'])
plt.show()