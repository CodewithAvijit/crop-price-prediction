import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector
data = pd.read_csv("processdata/CLEAN_CROP.csv")
print(data['District'].unique())
stateenconder = LabelEncoder()
districtenconder = LabelEncoder()
commodityenconder = LabelEncoder()
varietyenconder = LabelEncoder()
data['State']=stateenconder.fit_transform(data['State'])
data['District']=districtenconder.fit_transform(data['District'])
data['Commodity']=commodityenconder.fit_transform(data['Commodity'])
data['Variety']=varietyenconder.fit_transform(data['Variety'])
joblib.dump(stateenconder, "traindata/stateenconder.pkl")
joblib.dump(districtenconder, "traindata/districtenconder.pkl")
joblib.dump(commodityenconder, "traindata/commodityenconder.pkl")
joblib.dump(varietyenconder, "traindata/varietyenconder.pkl")
print(data.columns.to_list())
x=data[['Commodity','Variety','Grade_FAQ', 'Grade_Small','State','District','Min_x0020_Price', 'Max_x0020_Price' ]]
y=data['Modal_x0020_Price']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
r2 = r2_score(ytest, ypred) * 100  
mse = mean_squared_error(ytest, ypred)  
print(r2_score(ytest,lr.predict(xtest))*100)
print(mean_squared_error(ytest, lr.predict(xtest)))
joblib.dump(lr, "traindata/model.pkl")
