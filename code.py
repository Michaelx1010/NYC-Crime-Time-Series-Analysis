###### Data Cleaning ########

import pandas as pd
from datetime import datetime
from plotly import express as px

# Read the data
data = pd.read_csv("./dataset/crimedata_NYC.csv")

# Convert datetime to date
data['date_single'] = pd.to_datetime(data['date_single']).dt.date
violent_crimes = ["rape (except statutory rape)", "personal robbery", "aggravated assault", 
                  "murder and nonnegligent manslaughter", "arson", "kidnapping/abduction"]

violent_crime = data[data['offense_type'].isin(violent_crimes)]

# Sum up the counts by date
violent_crime_total = violent_crime.groupby('date_single').size().reset_index(name='Count')

# Sum up the counts by date for each category
violent_crime_t = violent_crime[['offense_type', 'date_single']]
violent_crime_t = violent_crime_t.groupby(['date_single', 'offense_type']).size().reset_index(name='Total_Crimes')
violent_crime_t['offense_type'] = violent_crime_t['offense_type'].map({
    "kidnapping/abduction": "kidnap",
    "rape (except statutory rape)": "rape",
    "murder and nonnegligent manslaughter": "murder",
    "personal robbery": "robbery"
}).fillna(violent_crime_t['offense_type'])
violent_crime_total['offense_type'] = "total"
violent_crime_total.columns = ['date_single', 'Total_Crimes']

# Combine total
v_crime = pd.concat([violent_crime_t, violent_crime_total], ignore_index=True)
v_crime = v_crime.sort_values(by=['date_single', 'offense_type'])
print(v_crime.head())

# Save the result to CSV file
# v_crime.to_csv("./dataset/crimedata_total.csv", index=False)
data_t = data[data['offense_type'] == "total"]

# Group by week and calculate total crimes
data_w = data_t.copy()
data_w['date_single'] = pd.to_datetime(data_w['date_single'])
data_w['week'] = data_w['date_single'].dt.to_period('W').apply(lambda r: r.start_time)
data_w = data_w.groupby(['week', 'offense_type']).size().reset_index(name='Total_Crimes')

# Save the result to CSV file
# data_w.to_csv("./dataset/crimedata_week.csv", index=False)

######## Data Preprocessing for RNN##########

## Parts of the code was adopted from Professor James' notes

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,mean_absolute_error
from math import sqrt
import seaborn as sns

# Read the crime data
df = pd.read_csv("./dataset/crimedata_week.csv")
df.drop('offense_type', axis = 1)
print(df.head())

# Redefine columns
df = df.rename(columns={"week": "t", "Total_Crimes": "y"})
df = df[["t","y"]]

t=np.array([*range(0,df.shape[0])])
x=np.array(df['y']).reshape(t.shape[0],1)
feature_columns=[0] # columns to use as features
target_columns=[0]  # columns to use as targets

# Visualize data
fig, ax = plt.subplots()
for i in range(0,x.shape[1]):
    ax.plot(t, x[:,i],'o',alpha = 0.5)
    ax.plot(t, x[:,i],"-")
ax.plot(t, 0*x[:,0],"-") 
ax.set_title("Raw Violent Crime Data")  
ax.set_xlabel("Time")                  
ax.set_ylabel("Count")  
plt.show()

x=(x-np.mean(x,axis=0))/np.std(x,axis=0)

# visualize normalized data 
fig, ax = plt.subplots()
for i in range(0,x.shape[1]):
    ax.plot(t, x[:,i],'o')
    ax.plot(t, x[:,i],"-")
ax.plot(t, 0*x[:,0],"-") 
ax.set_title("Normalized Violent Crime Data")  
ax.set_xlabel("Time")                  
ax.set_ylabel("Count") 
plt.show()


split_fraction = 0.8
cut = int(split_fraction * x.shape[0])
tt = t[0:cut]; xt = x[0:cut]
tv = t[cut:]; xv = x[cut:]

fig, ax = plt.subplots()
for i in range(0, x.shape[1]):
    ax.plot(tt, xt[:,i], 'ro', alpha=0.25, label="Train" if i == 0 else "")
    ax.plot(tt, xt[:,i], "g-")
for i in range(0, x.shape[1]):
    ax.plot(tv, xv[:,i], 'bo', alpha=0.25, label="Test" if i == 0 else "")
    ax.plot(tv, xv[:,i], "g-")

ax.set_title("Normalized Violent Crime Data (Train/Test Split)")  
ax.set_xlabel("Time")                  
ax.set_ylabel("Count") 

handles, labels = ax.get_legend_handles_labels()
# We filter handles and labels to remove duplicates
unique_labels = list(dict.fromkeys(labels))  # Using dict to preserve order
unique_handles = [handles[labels.index(label)] for label in unique_labels]
ax.legend(unique_handles, unique_labels, loc="best")

plt.show()

# Time series mini-batch function
def form_arrays(x,lookback=3,delay=1,step=1,feature_columns=[0],target_columns=[0],unique=False,verbose=False):
    # verbose=True --> report and plot for debugging
    # unique=True --> don't re-sample: 
    # x1,x2,x3 --> x4 then x4,x5,x6 --> x7 instead of x2,x3,x4 --> x5

    # initialize 
    i_start=0; count=0; 
    
    # initialize output arrays with samples 
    x_out=[]
    y_out=[]
    
    # sequentially build mini-batch samples
    while i_start+lookback+delay< x.shape[0]:
        
        # define index bounds
        i_stop=i_start+lookback
        i_pred=i_stop+delay
        
        # report if desired 
        if verbose and count<2: print("indice range:",i_start,i_stop,"-->",i_pred)

        # define arrays: 
        # method-1: buggy due to indexing from left 
        # numpy's slicing --> start:stop:step
        # xtmp=x[i_start:i_stop+1:steps]
        
        # method-2: non-vectorized but cleaner
        indices_to_keep=[]; j=i_stop
        while  j>=i_start:
            indices_to_keep.append(j)
            j=j-step

        # create mini-batch sample
        xtmp=x[indices_to_keep,:]    # isolate relevant indices
        xtmp=xtmp[:,feature_columns] # isolate desire features
        ytmp=x[i_pred,target_columns]
        x_out.append(xtmp); y_out.append(ytmp); 
        
        # report if desired 
        if verbose and count<2: print(xtmp, "-->",ytmp)
        if verbose and count<2: print("shape:",xtmp.shape, "-->",ytmp.shape)

        # PLOT FIRST SAMPLE IF DESIRED FOR DEBUGGING    
        if verbose and count<2:
            fig, ax = plt.subplots()
            ax.plot(x,'b-')
            ax.plot(x,'bx')
            ax.plot(indices_to_keep,xtmp,'go')
            ax.plot(i_pred*np.ones(len(target_columns)),ytmp,'ro')
            plt.show()
            
        # UPDATE START POINT 
        if unique: i_start+=lookback 
        i_start+=1; count+=1
        
    return np.array(x_out),np.array(y_out)

# training
L=25; S=1; D=1
Xt,Yt=form_arrays(xt,lookback=L,delay=D,step=S,feature_columns=feature_columns,target_columns=target_columns,unique=False,verbose=True)

# validation
Xv,Yv=form_arrays(xv,lookback=L,delay=D,step=S,feature_columns=feature_columns,target_columns=target_columns,unique=False,verbose=True)



##### Utility function for plots and reports ########
def history_plot(history):
    FS=18   #FONT SIZE
    # PLOTTING THE TRAINING AND VALIDATION LOSS 
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# UTILITY FUNCTION
def regression_report(yt,ytp,yv,yvp):
    print("---------- Regression report ----------")
    
    print("TRAINING:")
    print(" MSE:",mean_squared_error(yt,ytp))
    print(" MAE:",mean_absolute_error(yt,ytp))
    # print(" MAPE:",mean_absolute_percentage_error(Yt,Ytp))
    
    # PARITY PLOT
    fig, ax = plt.subplots()
    ax.plot(yt,ytp,'ro')
    ax.plot(yt,yt,'b-')
    ax.set(xlabel='y_data', ylabel='y_predicted',
        title='Training data parity plot (line y=x represents a perfect fit)')
    plt.show()
    
    # PLOT PART OF THE PREDICTED TIME-SERIES
    frac_plot=1.0
    upper=int(frac_plot*yt.shape[0]); 
    # print(int(0.5*yt.shape[0]))
    fig, ax = plt.subplots()
    ax.plot(yt[0:upper],'b-')
    ax.plot(ytp[0:upper],'r-',alpha=0.5)
    ax.plot(ytp[0:upper],'ro',alpha=0.25)
    ax.set(xlabel='index', ylabel='y(t (blue=actual & red=prediction)', title='Training: Time-series prediction')
    plt.show()

      
    print("VALIDATION:")
    print(" MSE:",mean_squared_error(yv,yvp))
    print(" MAE:",mean_absolute_error(yv,yvp))
    # print(" MAPE:",mean_absolute_percentage_error(Yt,Ytp))
    
    # PARITY PLOT 
    fig, ax = plt.subplots()
    ax.plot(yv,yvp,'ro')
    ax.plot(yv,yv,'b-')
    ax.set(xlabel='y_data', ylabel='y_predicted',
        title='Validation data parity plot (line y=x represents a perfect fit)')
    plt.show()
    
    # PLOT PART OF THE PREDICTED TIME-SERIES
    upper=int(frac_plot*yv.shape[0])
    fig, ax = plt.subplots()
    ax.plot(yv[0:upper],'b-')
    ax.plot(yvp[0:upper],'r-',alpha=0.5)
    ax.plot(yvp[0:upper],'ro',alpha=0.25)
    ax.set(xlabel='index', ylabel='y(t) (blue=actual & red=prediction)', title='Validation: Time-series prediction')
    plt.show()

sns.set(style="whitegrid")

def history_plot(history):
    FS = 18  # Font size
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=epochs, y=loss_values, marker='o', label='Training loss')
    sns.lineplot(x=epochs, y=val_loss_values, marker='o', label='Validation loss')
    plt.title('Training and Validation Loss', fontsize=FS)
    plt.xlabel('Epochs', fontsize=FS)
    plt.ylabel('Loss', fontsize=FS)
    plt.legend()
    plt.show()

def regression_report(yt, ytp, yv, yvp):
    FS = 18  # Font size for plots
    sns.set(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1.1, color_codes=True, rc=None)

    print("---------- Regression Report ----------")
    
    print("TRAINING:")
    print(" MSE:", mean_squared_error(yt, ytp))
    print(" MAE:", mean_absolute_error(yt, ytp))
    
    yt_flat = yt.flatten() if yt.ndim > 1 else yt
    ytp_flat = ytp.flatten() if ytp.ndim > 1 else ytp
    yv_flat = yv.flatten() if yv.ndim > 1 else yv
    yvp_flat = yvp.flatten() if yvp.ndim > 1 else yvp

    # Training Parity Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=yt_flat, y=ytp_flat, color='magenta', alpha=0.6, edgecolor='black', marker='o')
    sns.lineplot(x=yt_flat, y=yt_flat, color='darkblue', label="y=x (Perfect fit)", linewidth=2.5)
    plt.title('Training Data Parity Plot', fontsize=FS)
    plt.xlabel('Actual Values', fontsize=FS)
    plt.ylabel('Predicted Values', fontsize=FS)
    plt.legend()
    plt.show()
    
    # Plot part of the Predicted Time-Series for Training
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=np.arange(len(yt_flat)), y=yt_flat, label='Actual', color='blue', linewidth=1.2)
    sns.lineplot(x=np.arange(len(ytp_flat)), y=ytp_flat, label='Predicted', color='red', alpha=0.75, linewidth=1.2)
    plt.title('Training: Time-series Prediction', fontsize=FS)
    plt.xlabel('Time Index', fontsize=FS)
    plt.ylabel('Target Variable', fontsize=FS)
    plt.legend()
    plt.show()
    
    print("VALIDATION:")
    print(" MSE:", mean_squared_error(yv, yvp))
    print(" MAE:", mean_absolute_error(yv, yvp))
    
    # Validation Parity Plot
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x=yv_flat, y=yvp_flat, color='magenta', alpha=0.6, edgecolor='black', marker='o')
    sns.lineplot(x=yv_flat, y=yv_flat, color='darkblue', label="y=x (Perfect fit)", linewidth=2.5)
    plt.title('Validation Data Parity Plot', fontsize=FS)
    plt.xlabel('Actual Values', fontsize=FS)
    plt.ylabel('Predicted Values', fontsize=FS)
    plt.legend()
    plt.show()
    
    # Plot part of the Predicted Time-Series for Validation
    plt.figure(figsize=(12, 7))
    sns.lineplot(x=np.arange(len(yv_flat)), y=yv_flat, label='Actual', color='blue', linewidth=2.5)
    sns.lineplot(x=np.arange(len(yvp_flat)), y=yvp_flat, label='Predicted', color='red', alpha=0.75, linewidth=2.5)
    plt.title('Validation: Time-series Prediction', fontsize=FS)
    plt.xlabel('Time Index', fontsize=FS)
    plt.ylabel('Target Variable', fontsize=FS)
    plt.legend()
    plt.show()
    
    
###########  RNN CODE ########

# Please note some parts of the code are not included in presentations and reports due to low performance.

###### Simple RNN #####

# RESHAPE INTO A DATA FRAME 
Xt1=Xt.reshape(Xt.shape[0],Xt.shape[1]*Xt.shape[2])
Xv1=Xv.reshape(Xv.shape[0],Xv.shape[1]*Xv.shape[2])

# # HYPERPARAMETERS 
optimizer="rmsprop"
loss_function="MeanSquaredError" 
learning_rate=0.001
numbers_epochs=200 #100
L2=1e-4
input_shape=(Xt.shape[1],Xt.shape[2])

# # batch_size=1                       # stocastic training
#batch_size=int(len(Xt1)/2.)    # mini-batch training
#batch_size = int(len(Xt1) * 0.1)
batch_size=len(Xt1)              # batch training

# BUILD MODEL
recurrent_hidden_units=64

# CREATE MODEL
model = keras.Sequential()

# ADD RECURRENT LAYER

# #COMMENT/UNCOMMENT TO USE RNN, LSTM,GRU
#model.add(LSTM(
#model.add(GRU(
model.add(SimpleRNN(
units=recurrent_hidden_units,
return_sequences=False,
input_shape=input_shape, 
recurrent_regularizer=regularizers.L2(L2),
#recurrent_dropout=0.8,
activation='tanh')
          ) 
     
# NEED TO TAKE THE OUTPUT RNN AND CONVERT TO SCALAR 
model.add(Dense(units=1, activation='linear'))

# MODEL SUMMARY
print(model.summary()); #print(x_train.shape,y_train.shape)
# # print("initial parameters:", model.get_weights())

# # COMPILING THE MODEL 
opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
#opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
# opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss=loss_function)

# TRAINING YOUR MODEL
history = model.fit(Xt,
                    Yt,
                    epochs=numbers_epochs,
                    batch_size=batch_size, verbose=False,
                    validation_data=(Xv, Yv))
# History plot
history_plot(history)

# Predictions 
Ytp=model.predict(Xt)
Yvp=model.predict(Xv) 

# REPORT
regression_report(Yt,Ytp,Yv,Yvp) 

RNN_e_r = sqrt(mean_squared_error(Yt, Ytp))



##### Multi-layer simpler RNN ######
optimizer = "rmsprop"
loss_function = "MeanSquaredError"
learning_rate = 0.001
numbers_epochs = 200
L2 = 1e-4
batch_size = len(Xt) 
input_shape = (Xt.shape[1], Xt.shape[2])
recurrent_hidden_units = 64

# Build a multi-layer Simple RNN model
model = keras.Sequential([
    SimpleRNN(recurrent_hidden_units, return_sequences=True, input_shape=input_shape,
              recurrent_regularizer=regularizers.L2(L2), activation='tanh'),
    SimpleRNN(recurrent_hidden_units, return_sequences=False, 
              recurrent_regularizer=regularizers.L2(L2), activation='tanh'),
    Dense(1, activation='linear')
])

print(model.summary())

# Compile the model 
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate), loss=loss_function)

# Train the model
history = model.fit(Xt, Yt, epochs=numbers_epochs, batch_size=batch_size, verbose=1, validation_data=(Xv, Yv))
history_plot(history)
Ytp = model.predict(Xt)
Yvp = model.predict(Xv)
RNN_e_r = sqrt(mean_squared_error(Yt, Ytp))
regression_report(Yt, Ytp, Yv, Yvp)



######### GRUs #########




# # HYPERPARAMETERS 
optimizer="rmsprop"
loss_function="MeanSquaredError" 
learning_rate=0.001
numbers_epochs=200 #100
L2=1e-4
input_shape=(Xt.shape[1],Xt.shape[2])

# # batch_size=1                       # stocastic training
batch_size=int(len(Xt1)/2.)    # mini-batch training
#batch_size = int(len(Xt1) * 0.1)
# batch_size=len(Xt1)              # batch training

# BUILD MODEL
recurrent_hidden_units=64

# CREATE MODEL
model = keras.Sequential()

# ADD RECURRENT LAYER

# #COMMENT/UNCOMMENT TO USE RNN, LSTM,GRU
#model.add(LSTM(
model.add(GRU(
#model.add(SimpleRNN(
units=recurrent_hidden_units,
return_sequences=False,
input_shape=input_shape, 
recurrent_regularizer=regularizers.L2(L2),
# recurrent_dropout=0.8,
activation='relu')
          ) 
     
# NEED TO TAKE THE OUTPUT RNN AND CONVERT TO SCALAR 
model.add(Dense(units=1, activation='linear'))

# MODEL SUMMARY
print(model.summary()); #print(x_train.shape,y_train.shape)
# # print("initial parameters:", model.get_weights())

# # COMPILING THE MODEL 
opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=loss_function)

# TRAINING YOUR MODEL
history = model.fit(Xt,
                    Yt,
                    epochs=numbers_epochs,
                    batch_size=batch_size, verbose=False,
                    validation_data=(Xv, Yv))
# History plot
history_plot(history)

# Predictions 
Ytp=model.predict(Xt)
Yvp=model.predict(Xv) 

# REPORT
regression_report(Yt,Ytp,Yv,Yvp) 
GRU_e_r = sqrt(mean_squared_error(Yt, Ytp))








######### GRU with multiple layers #########


# Set hyperparameters
optimizer = "rmsprop"
loss_function = "MeanSquaredError"
learning_rate = 0.001
numbers_epochs = 200
L2 = 1e-4
batch_size = int(len(Xt) / 2.) 
input_shape = (Xt.shape[1], Xt.shape[2])
recurrent_hidden_units = 64

# Build the multi-layer GRU model
model = keras.Sequential([
    GRU(recurrent_hidden_units, return_sequences=True, input_shape=input_shape,
        recurrent_regularizer=regularizers.L2(L2), activation='relu'),
    GRU(recurrent_hidden_units, return_sequences=False,
        recurrent_regularizer=regularizers.L2(L2), activation='tanh'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate), loss=loss_function)

# Train the model
history = model.fit(Xt, Yt, epochs=numbers_epochs, batch_size=batch_size, validation_data=(Xv, Yv))


history_plot(history)

Ytp = model.predict(Xt)
Yvp = model.predict(Xv)
GRU_e_r = sqrt(mean_squared_error(Yt, Ytp))
regression_report(Yt, Ytp, Yv, Yvp)





############## LSTM #############




# # HYPERPARAMETERS 
optimizer="rmsprop"
loss_function="MeanSquaredError" 
learning_rate=0.001
numbers_epochs=200 #100
L2=1e-4
input_shape=(Xt.shape[1],Xt.shape[2])

# # batch_size=1                       # stocastic training
# # batch_size=int(len(x_train)/2.)    # mini-batch training
batch_size=len(Xt1)              # batch training

# BUILD MODEL
recurrent_hidden_units=64

# CREATE MODEL
model = keras.Sequential()

# ADD RECURRENT LAYER

# #COMMENT/UNCOMMENT TO USE RNN, LSTM,GRU
model.add(LSTM(
#model.add(GRU(
#model.add(SimpleRNN(
units=recurrent_hidden_units,
return_sequences=False,
input_shape=input_shape, 
recurrent_regularizer=regularizers.L2(L2),
recurrent_dropout=0.8,
activation='tanh')
          ) 
     
# NEED TO TAKE THE OUTPUT RNN AND CONVERT TO SCALAR 
model.add(Dense(units=1, activation='linear'))

# MODEL SUMMARY
print(model.summary()); #print(x_train.shape,y_train.shape)
# # print("initial parameters:", model.get_weights())

# # COMPILING THE MODEL 
opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(optimizer=opt, loss=loss_function)

# TRAINING YOUR MODEL
history = model.fit(Xt,
                    Yt,
                    epochs=numbers_epochs,
                    batch_size=batch_size, verbose=False,
                    validation_data=(Xv, Yv))
# History plot
history_plot(history)

# Predictions 
Ytp=model.predict(Xt)
Yvp=model.predict(Xv) 

# REPORT
regression_report(Yt,Ytp,Yv,Yvp) 
LSTM_e_r = sqrt(mean_squared_error(Yt, Ytp))




######## BiLSTM ###########


input_shape = (Xt.shape[1], Xt.shape[2]) 

# Define the model architecture
model_biLSTM = keras.Sequential([
    Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
    Dense(1, activation='linear')
])

# Compile the model
model_biLSTM.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mean_squared_error')

# Display the model summary to verify its structure
print(model_biLSTM.summary())

# Train the model
history_biLSTM = model_biLSTM.fit(Xt, Yt, epochs=200, batch_size=len(Xt), verbose=1, validation_data=(Xv, Yv))

# Plot training and validation loss
history_plot(history_biLSTM)

# Predict and evaluate the model
Yt_pred_biLSTM = model_biLSTM.predict(Xt)
Yv_pred_biLSTM = model_biLSTM.predict(Xv)
regression_report(Yt, Yt_pred_biLSTM, Yv, Yv_pred_biLSTM)



######## Model Comparison ########
import seaborn as sns
# Creating a DataFrame to hold the data
data = {
    'Model': ['RNN', 'GRU', 'LSTM', 'BiLSTM'],
    'RMSE': [ RNN_e_r,  GRU_e_r, LSTM_e_r, Bi]
}

df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Set the palette to have different colors for each bar
palette = sns.color_palette("hsv", len(df))

# Plotting using seaborn
plt.figure(figsize=(10,6))
sns.barplot(y='RMSE', x='Model', data=df, palette=palette)
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('Comparison of Deep Learning Models')
plt.show()


###### Code for drawing model architecture diagram #######

import matplotlib.pyplot as plt

def draw_model_diagram(ax, model_type='SimpleRNN', units=64, activation='tanh'):
    # Drawing the model layout
    if model_type == 'GRU':
        hidden_activation = 'tanh & sigmoid'  
    elif model_type == 'Bi-directional LSTM':
        hidden_activation = 'tanh & sigmoid'  
    else:
        hidden_activation = 'tanh'  

    ax.text(0.5, 0.7, 'Input Layer\n(batch, timesteps, features)', ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='b', facecolor='w'))
    ax.text(0.5, 0.5, f'{model_type}\n{units} units\n({hidden_activation})', ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='r', facecolor='w'))
    ax.text(0.5, 0.2, 'Dense Layer\n1 unit (linear)', ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='g', facecolor='w'))
    
    # Drawing arrows
    ax.arrow(0.5, 0.65, 0, -0.1, head_width=0.05, head_length=0.05, fc='k', ec='k')
    ax.arrow(0.5, 0.4, 0, -0.15, head_width=0.05, head_length=0.05, fc='k', ec='k')
    
    # Setting limits and cleaning up axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
draw_model_diagram(axs[0, 0], 'SimpleRNN', 64, 'tanh')
axs[0, 0].set_title('Simple RNN Architecture')
draw_model_diagram(axs[0, 1], 'GRU', 64, 'tanh & sigmoid')
axs[0, 1].set_title('GRU Architecture')
draw_model_diagram(axs[1, 0], 'LSTM', 64, 'tanh & sigmoid')
axs[1, 0].set_title('LSTM Architecture')
draw_model_diagram(axs[1, 1], 'Bi-directional LSTM', 128, 'tanh & sigmoid')
axs[1, 1].set_title('Bi-directional LSTM Architecture')

plt.tight_layout()


######################  CNN ##############
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(Xt.shape[1], Xt.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(1)
])

# Compile the model
model_cnn.compile(optimizer='adam', loss='mean_squared_error')

# Model summary
print(model_cnn.summary())
history = model_cnn.fit(Xt, Yt, epochs=100, batch_size=32, validation_data=(Xv, Yv))

history_plot(history)

validation_loss = model_cnn.evaluate(Xv, Yv, verbose=0)
print(f"Validation Loss: {validation_loss}")

# Predictions 
Ytp=model_cnn.predict(Xt)
Yvp=model_cnn.predict(Xv) 

# REPORT
regression_report(Yt,Ytp,Yv,Yvp)