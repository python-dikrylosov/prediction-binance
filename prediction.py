import time, math
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
real_date = time.strftime("%Y-%m-%d")
data_btcusdt = yf.download("AXS-BTC", start="2014-04-01", end=real_date, interval='1d')
print(data_btcusdt)
# create dATEFRAME CLOSE
data = data_btcusdt.filter(['Close'])
print(data)
print(data.shape)
# convert dataframe
dataset = data.values
print(dataset)
# get the number rows to train the model
training_data_len = math.ceil(len(dataset) * .8)
print(training_data_len)
# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)
# create the training dataset
train_data = scaled_data[0:training_data_len, :]
print(train_data)
# split the data into x_train and y_train data sets
x_train = []
y_train = []
for rar in range(60, len(train_data)):
    x_train.append(train_data[rar - 60:rar, 0])
    y_train.append(train_data[rar, 0])
    if rar <= 61:
        print(x_train)
        print(y_train)
        print()
# conver the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
# reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)
# biuld to LST model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(101, return_sequences=False))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(1))
# cmopale th emodel
model.compile(optimizer='adam', loss='mean_squared_error')
# train_the_model
model.summary()
print("Fit model on training data")
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_train, y_train, batch_size=1)
print("test loss, test acc:", results)
#filename = "BTC-USDT_exterminate 2018-02-01_1D.h5"
#model = tf.keras.models.load_model(filename)
model.fit(x_train, y_train, batch_size=1, epochs=1)
#model.save(filename)
#reconstructed_model = tf.keras.models.load_model(filename)
#np.testing.assert_allclose(model.predict(x_train), reconstructed_model.predict(x_train))
#reconstructed_model.fit(x_train, y_train)
# create the testing data set
# create a new array containing scaled values from index 1713 to 2216
test_data = scaled_data[training_data_len - 60:, :]
# create the fata sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for resr in range(60, len(test_data)):
  x_test.append(test_data[resr - 60:resr, 0])
# conert the data to numpy array
x_test = np.array(x_test)
# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# get the root squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)
# get the quate
btc_quote = yf.download("AXS-BTC", start="2014-04-01", end=real_date, interval='1d')
# btc_quote = pd.read_csv(str(balance_btc["asset"]) + ".csv", delimiter=",")
# new_df = btc_quote.filter(["Well"])
new_df = btc_quote.filter(['Close'])

# get teh last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
# scale the data to be values beatwet 0 and 1

last_60_days_scaled = scaler.transform(last_60_days)

# creAte an enemy list
X_test = []
# Append past 60 days
X_test.append(last_60_days_scaled)

# convert the x tesst dataset to numpy
X_test = np.array(X_test)

# Reshape the dataframe
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# get predict scaled

pred_price = model.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)
import numpy as np
pred_price_a = pred_price[0]
pred_price_aa = pred_price_a[0]
preset_pred_price = float(pred_price_aa)
print(pred_price)
print(preset_pred_price)
