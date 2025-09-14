import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('house_price_dataset.csv')

x=data.drop(columns='price',axis=1)
y=data['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()

rfe=RFE(estimator=model,n_features_to_select=3)
rfe=rfe.fit(x_train,y_train)

selected_features=x_train.columns[rfe.support_]

x_train_rfe=x_train[selected_features]
x_test_rfe=x_test[selected_features]

model.fit(x_train_rfe,y_train)

y_pred=model.predict(x_test_rfe)

mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
# print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")