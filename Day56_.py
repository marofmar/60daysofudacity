'''
Day 56
- Download JavaEE for JSP programming study
- Download apache tomcat and learned how to run and shutdown that on my machine
- Also learned that .bat is for Windows and .sh for Mac
	> in cmd go to the downloaded folder
	> chmod +x *.sh
	> ./startup.sh 
	> ./shutdown.sh
	> check on localhost:8080
- Added apache tomcat to Eclipse EE
- Created one Dynamic Web Project on eclipse
- Wrote a simple file and exported WAR file to Tomcat

- Python energy prediction program code blocks
'''

# 1) upload local files in Colab environment
from google.colab import files
upload1 = files.upload()

# 2) load dependencies and set parameters
SEQ_LEN = 1
FUTURE_PERIOD_PREDICT = 1

# Parameters setting 
#========================================
BATCH_SIZE = 64
TITLE = 'Independence'
l1 = 8
l2 = 32
#index_n = -3 

#plust one 
date_MDDYYYYVTTTT = '8/16/2019 00:00' 
# 시작 및 끝 추가 테스트용
test_start = '8/14/2019 00:00'
test_end = '8/16/2019 23:00'
#========================================
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque

from sklearn import model_selection

import random
from time import gmtime, strftime
from datetime import datetime
import itertools
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM,Activation, Dropout, BatchNormalization

import matplotlib.pyplot as plt
import seaborn as sns

# 3) read the file
import io
import pandas as pd 
dataset = pd.read_excel(io.BytesIO(upload1['evergy_v10.xlsx']), index_col = 0)

# 4) Select columns that I am interested in to analyze
d2 = ['ss_all','month', 'date','hour','day','elec']
#data = dataset.copy()
dt = data[d2]

# 5) Dummify some of categorial data
# dt = pd.get_dummies(dt, columns=['month', 'date','hour','day'])
dt = pd.get_dummies(dt, columns=['month', 'date','hour','day'])
display(dt.head(2))
display(dt.tail(2))
save_dt = dt.copy()

# 6) Make sure the numbers are in the float datatype. Sometiems they were read in string type.
def to_float(dt):
  pm3 = dt
  iter_cols = ['elec']
  for i in iter_cols:
    pm3[i] = pm3[i].astype(float)
  return pm3 

dt = to_float(dt)

# 7) Normalize numeric data "Preprocessing"
def norm(dt): #기온이랑 강수 no들어감 
  pm3 = dt
  scaler = preprocessing.MinMaxScaler()
  pm3['elec'] = pm3['elec']/float(100000.0)
  #pm3['shift_1'] = pm3['elec'].shift(24)
  pm3['shift_7'] = pm3['elec'].shift(168)
  #pm3['temp_tt'] = scaler.fit_transform(np.array(pm3['temp_tt']).reshape(-1,1))
  #pm3['rainfall_tt'] = scaler.fit_transform(np.array(pm3['rainfall_tt']).reshape(-1,1))
  #dt.dropna(inplace = True)
  return pm3

dt = norm(dt)
display(dt.head(2))
display(dt.tail(2))


# 8) Re-order the column data, since my target data is Elec, I put that in the very right side
def reorder(dt):
  
  cols = dt.columns.tolist()
  cols.remove('elec')
  cols.append('elec')
  print(cols)
  dt = dt[cols]
  return dt

dt= reorder(dt)

# 9) Split out test data from the preprocessed data
test = dt.loc[test_start: test_end]
train = dt.loc['1/08/2000 00:00':'8/11/2019 23:00']

# 10) Also split out validation dataset, for the two week length
def tr_val_te_one(pm3, val_days = 14):  
  tr = pm3.iloc[:-24*val_days,:]
  #tr = tr.dropna(inplace = True)
  val = pm3.iloc[-24*val_days:,:] # 2주 정도 이전의 동일 요일, 시간 등 정보를 통해 검증할 수 있도록 
  #te = pm3.iloc[-24:,:]
  return tr, val

tr, val = tr_val_te_one(train)
#train = train.dropna(inplace = True)
# display(train.head(3))
# display(train.tail(3))
print(train)

# 11) Now, need to split feature dataset X and target data y 
def _deque_df(df):
    

    sequential_data = []
    prev_points = deque(maxlen = SEQ_LEN)

    for i in df.values:
        prev_points.append([n for n in i[:-1]])
        if len(prev_points) == SEQ_LEN:
            sequential_data.append([np.array(prev_points), i[-1]])

    #random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)



train_X, train_y = _deque_df(tr)
validation_X, validation_y = _deque_df(val)
test_X, test_y = _deque_df(test)

# 12) Time to reshape the splitted data, so that the model can accept them smoothely
def to_reshape_X(train_X, validation_X, test_X):
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[-1])
    validation_X = validation_X.reshape(validation_X.shape[0],validation_X.shape[-1])
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[-1])
    return train_X, validation_X, test_X

#위에서 만든 함수 바로 씀, 2차원으로 쉐이프 줄여주는 것 
train_X, validation_X, test_X = to_reshape_X(train_X, validation_X, test_X)
print(train_X.shape, validation_X.shape, test_X.shape)

# 13) Build a neural network model
def to_model(l1, l2):
  
  model = Sequential() # keras, default: glorot_uniform, zeros
  model.add(Dense(l1, activation='elu', input_dim = train_X.shape[1]))
  model.add(Dense(l2, activation='elu', input_dim=l1))
  model.add(Dense(1, activation='elu'))
  print(model.summary())

  #opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
  #model.compile(loss = 'mse', optimizer = opt, metrics = ['accuracy'])
  model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['accuracy'])
  return model

  model = to_model(l1,l2)

# 14) Fit the data into the model 
def to_fit_model(model,train_X , train_y, validation_X, validation_y, test_X, BATCH_SIZE, EPOCHS ):

model.fit(train_X, train_y, batch_size = BATCH_SIZE, epochs = EPOCHS,
       validation_data = (validation_X, validation_y))

pred = model.predict(test_X)
pred = list(itertools.chain(*pred))
return pred

# 15) Get the prediction from the model
pred = to_fit_model(model,train_X , train_y, validation_X, validation_y, test_X, BATCH_SIZE=64, EPOCHS=10 )
pred = np.asarray(pred) * 100000. # get the data scale back 

# 16) draw the graph
plt.figure(figsize=(20,8)) #figsize 라는 변수를 이용해서 그래프 사이즈 조정. 이 변수 없이 그냥 plt.figure() 만 입력하면 기본적으로 아래 크기의 한 1/4 정도만 출력됨
#plt.title("2019 Independence Day MAPE = {}".format(round(total_mape,2))) # title 은 그래프 제목 의미함. format 이라는 추가 함수를 이용해서 제목에 mape 값도 동시에 나오도록 함
plt.title('2019 Independence Day +- 1day')
plt.plot(pred, label='PRED', linewidth = 7.0) # 선 굵기는 linewidth 로 조정. 예측값은 앞서 만든 result 데이터프레임의 y_pred 칼럼 값 이용. 따라서 result.y_pred
#plt.plot(result.y_true, label='true', linestyle = '--', linewidth = 7.0)
plt.legend() #차트의 레전드 출력한다는 함수 입력.
plt.show() #완성된 차트를 보여주는 함수 



