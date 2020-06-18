import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model


loans = pd.read_csv('loans_clean.csv',index_col=0)



#Drop unnecesary variables from the model
loans.drop(['ZIP_3'],axis=1,inplace=True)


#cut and scale the data
def prep_data(df,cut,rand_seed):
    X = df.drop('Prep',axis=1).values
    y = df['Prep'].values
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=cut,random_state=rand_seed)
    # scale the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# define and fit the model
def get_model(X_train, y_train):
    #define the model
    model = Sequential()
    # input layer
    model.add(Dense(14, activation='relu'))
    model.add(Dropout(0.2))
    # hidden layer
    model.add(Dense(7, activation='relu'))
    model.add(Dropout(0.2))
    # output layer
    model.add(Dense(units=1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    #Prepare the early stop in case of overfitting
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    #We'll also add the Tensorflow parameters for the data visualization
    log_directory = 'logs\\fit'
    timestamp = datetime.now().strftime("%Y-%m-%d--%H%M")
    log_directory = log_directory + '\\' + timestamp
    board = TensorBoard(log_dir=log_directory, histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=1)
    #Now we fit the model
    model.fit(x=X_train,
          y=y_train,
          epochs=600,
          validation_data=(X_test, y_test), verbose=1,
          callbacks=[early_stop,board]
          )
    return model


# get the data for the model
X_train, X_test, y_train, y_test = prep_data(loans,.3,42)

# fit model
model = get_model(X_train, y_train)

#Save the model
model.save('loan_prep_model_v2.h5')

#predict the model
predictions = model.predict_classes(X_test)

#Now we get the metrics to evaluate our model

#Printing the Confusion Matrix
print(confusion_matrix(y_test,predictions))

#Printing Precision Score
print('Precision: %f' % average_precision_score(y_test, predictions))

#Printing Recall Score
print('Recall: %f' % recall_score(y_test, predictions))

#Printing area under ROC Curve
print('AUC: %f' % roc_auc_score(y_test, predictions))
