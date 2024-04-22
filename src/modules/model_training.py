import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import joblib as jb
from .surface_plots import surface_plots
from config import model_training as cfg
from config import paths

def model_training(mode='all'):

  # Load the training and validation data from csv files
  X_train = np.loadtxt(paths.processed_data / 'X_train.csv',
                       delimiter=",", skiprows=1)
  X_val = np.loadtxt(paths.processed_data / 'X_val.csv',
                     delimiter=",", skiprows=1)
  y_train = np.loadtxt(paths.processed_data / 'y_train.csv',
                       delimiter=",", skiprows=1)
  y_val = np.loadtxt(paths.processed_data / 'y_val.csv',
                     delimiter=",", skiprows=1)

  if (mode=='rf' or mode=='all'):
    # Start the random forest model training
    print('Starting Random Forest training...')
    rforest = RandomForestRegressor()
    train_rf_start = time.time()
    rforest.fit(X_train, y_train)
    train_rf_end = time.time()
    train_rf_time = (train_rf_end - train_rf_start)/60
    print('RF training time: {:.3g} minutes.'.format(train_rf_time))

    # Save the model
    jb.dump(rforest, paths.models / 'rf_model.joblib')
    print('Successfully completed Random Forest training.')

  if (mode=='nn' or mode=='all'):
    print('Starting Neural Network training...')

    # Set a random seed for tensorflow and numpy to ensure reproducibility
    tf.random.set_seed(10)
    np.random.seed(10)

    # Obtain the standard deviations of the training data
    dB_dt_std = np.std(y_train[:,0])
    dD_dt_std = np.std(y_train[:,1])

    #define a loss function
    def custom_mae(y_true, y_pred):
      loss = y_pred - y_true
      loss = loss / [dB_dt_std, dD_dt_std]
      loss = K.abs(loss)
      loss = K.sum(loss, axis=1) 
      return loss
    
    hp = cfg.get_nn_hp()

    # Tune the hyperparameters if specified
    if cfg.tuning_hp_vals:
      print('Starting hyperparameter tuning...')
      best_hp = hp_tuning([X_train, X_val, y_train, y_val], hp, custom_mae,
                          cfg.tuning_hp_name, cfg.tuning_hp_vals)
      hp[cfg.tuning_hp_name] = best_hp
      print('Successfully tuned hyperparameters.')

    # Define the model
    nnetwork = tf.keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun'],
                                        kernel_regularizer=tf.keras.regularizers.l1(hp['l1_reg'])))
    nnetwork.add(tf.keras.layers.Dense(2, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1(hp['l1_reg'])))
    # Compile and fit the model
    nnetwork.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=custom_mae)
    train_nn_start = time.time()
    history = nnetwork.fit(X_train, y_train, epochs = hp['n_epochs'], validation_data = (X_val, y_val),
                               batch_size = hp['batch_size'])
    
    # Save the history as a pandas dataframe
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(paths.outputs / 'training_history.csv')
    
    # Plot the MSE history of the training
    plt.figure()
    for key in history.history.keys():
      plt.plot(history_df[key], label=key)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE')
    plt.savefig(paths.figures / 'training_history.png')

    # Calculate the training time and save the model
    train_nn_end = time.time()
    train_nn_time = (train_nn_end - train_nn_start)/60
    print('NN training time: {:.3g} minutes.'.format(train_nn_time))

    nnetwork.save(paths.models / 'nn_model.h5')
    print('Successfully completed Neural Network training.')

    # Retrieve the loss name
    try:
      loss_name = nnetwork.loss.__name__
    except AttributeError:
      loss_name = nnetwork.loss.name

  # If training only one model, load the parameters from the other model
  if (mode=='rf' or mode=='nn'):
    with open(paths.temp_data / 'train_summary.pkl', 'rb') as f:
        rf_summary, nn_summary = pickle.load(f)
      
  # Save the training summary
  if (mode=='rf' or mode=='all'):
    rf_summary = "".join(['\n\n*MODEL TRAINING*',
                          '\n\nRANDOM FOREST:',
                          '\ntrain_rf_time = {:.1f} minutes'.format(train_rf_time),
                          '\nn_estimators = {}'.format(rforest.get_params()['n_estimators']),
                          '\nmax_features = {}'.format(rforest.get_params()['max_features']),
                          '\nmax_samples = {}'.format(rforest.get_params()['max_samples']),
                          '\nmin_samples_leaf = {}'.format(rforest.get_params()['min_samples_leaf']),
                          '\nmin_samples_split = {}'.format(rforest.get_params()['min_samples_split'])])
  if (mode=='nn' or mode=='all'):
    nn_summary = "".join(['\n\nNEURAL NETWORK:',
                          '\ntrain_nn_time = {:.1f} minutes'.format(train_nn_time),
                          '\nloss_name = {}'.format(loss_name)])

    for key, value in hp.items():
      nn_summary += f"\n{key} = {value}"

  with open(paths.temp_data / 'train_summary.pkl', 'wb') as f:
      pickle.dump([rf_summary, nn_summary], f)

  return

def hp_tuning(train_val_data, hp, loss, tuning_hp_name, tuning_hp_vals):

  # Unpack the data
  X_train, X_val, y_train, y_val = train_val_data

  # Create an empty list to store the losses
  losses = []

  # Make a mask to subset of the data for tuning
  tuning_mask = np.random.rand(len(X_train)) < cfg.tuning_size

  # Subset the data for tuning
  X_tuning = X_train[tuning_mask]
  y_tuning = y_train[tuning_mask]

  # Test each hyperparameter value
  for i, value in enumerate(tuning_hp_vals):
    print(f'Testing hp {i+1} of {len(tuning_hp_vals)}...')
    hp[tuning_hp_name] = value

    # Define the model
    nnetwork = tf.keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun'],
                                        kernel_regularizer=tf.keras.regularizers.l1(hp['l1_reg'])))
    nnetwork.add(tf.keras.layers.Dense(2, activation='linear',
                                    kernel_regularizer=tf.keras.regularizers.l1(hp['l1_reg'])))

    # Compile and fit the model
    nnetwork.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=loss)
    history = nnetwork.fit(X_tuning, y_tuning, epochs = hp['n_epochs'], validation_data = (X_val, y_val), 
                          batch_size = hp['batch_size'])
    
    # Make a surface plot of the model
    surface_plots(nnetwork, name=f'{tuning_hp_name}_{value}')

    losses.append(history.history['val_loss'][-1])

  # Save a csv with each model's loss and hp
  tuning_res_dict = {tuning_hp_name:tuning_hp_vals, 'loss':losses}
  pd.DataFrame(tuning_res_dict).to_csv(paths.outputs / 'hp_tuning.csv')

  # Select the best hyperparameter and retrain the model
  best_hp = tuning_hp_vals[np.argmin(losses)]
  print('Best model has {} = {:.1g}.'.format(tuning_hp_name, best_hp))
  return best_hp