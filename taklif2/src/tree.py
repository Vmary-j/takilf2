import pandas as pd


df = pd.read_csv("data/cleaned_data.csv", index_col=0, parse_dates=['datetime'])
df_hourly = pd.read_csv("data/df_hourly.csv", index_col=0, parse_dates=True)
df_daily = pd.read_csv("data/df_daily.csv", index_col=0, parse_dates=True)


df_hourly['lag_1'] = df_hourly['Global_active_power'].shift(1)
df_hourly.dropna(inplace=True)  


train_size = int(len(df_hourly) * 0.8)
train, test = df_hourly[:train_size], df_hourly[train_size:]


X_train, y_train = train[['lag_1']], train['Global_active_power']
X_test, y_test = test[['lag_1']], test['Global_active_power']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


X = df_hourly.dropna().drop(columns=['Global_active_power'])  
y = df_hourly.dropna()['Global_active_power']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


y_pred_lr = lr_model.predict(X_test)


mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = mean_squared_error(y_test, y_pred_lr, squared=False)
print(f'Linear Regression - MAE: {mae_lr:.2f}, RMSE: {rmse_lr:.2f}')


rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


y_pred_rf = rf_model.predict(X_test)


mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f'Random Forest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}')


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, label='True Values')
plt.plot(y_test.index, y_pred_rf, label='Predicted Values', linestyle='--')
plt.title('Random Forest - Forecasting')
plt.legend()

plt.tight_layout()
plt.show()