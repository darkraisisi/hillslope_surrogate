import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
import os
import joblib as jb
from .data_formatting import subset_mask_stratified

def train_models(train_val_data, add_train_vars=[None]*2,
                 mode='all'):

  # Unpack the data
  X_train, X_val, y_train, y_val = train_val_data
  w_train, len_eq_train = add_train_vars

    # # Shuffle the training data
    # shuffled_indices = np.arange(len(X_train))
    # np.random.shuffle(shuffled_indices)
    # X_train = np.squeeze(X_train[shuffled_indices])
    # y_train = np.squeeze(y_train[shuffled_indices])
    # w_train = np.squeeze(w_train[shuffled_indices])

  # Set the percentage of linear data epochs/trees to use for training
  pct_lin = 0.25

  if (mode=='rf' or mode=='all'):
    # Start the random forest model training
    print('Starting Random Forest training...')
    n_estimators = 200

    rforest = RandomForestRegressor(n_estimators = n_estimators,
                                    max_features = 'sqrt',
                                    max_samples = 0.4,
                                    min_samples_leaf = 2,
                                    min_samples_split = 20)
    train_rf_start = time.time()
    rforest.fit(X_train, y_train, sample_weight=w_train)

    train_rf_end = time.time()
    train_rf_time = (train_rf_end - train_rf_start)/60
    print('RF training time: {:.3g} minutes.'.format(train_rf_time))

    # Save the model
    jb.dump(rforest, os.path.join('data','rf_model.joblib')) 
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

    # Set the hyperparameters
    hp = {'units':[9, 27, 81, 162, 324, 648, 1296], 'act_fun':'relu',
          'learning_rate':1E-5, 'batch_size':128, 'l1_reg':1e-5, 'n_epochs':200}

    # Define what hyperparameter to tune and its values
    tuning_hp_name = 'w_eq'
    tuning_hp_vals = [i/10 for i in range(1,10)]

    if tuning_hp_vals:
      print('Starting hyperparameter tuning...')
      best_hp = hp_tuning(train_val_data, add_train_vars, hp, custom_mae,
                          tuning_hp_name, tuning_hp_vals)
      # Apply the best hyperparameter
      if tuning_hp_name == 'w_eq':
        # Generate the weights for the equilibrium and jumps data
        length_ratio = len_eq_train/(len(X_train) - len_eq_train)
        w_train = np.concatenate((best_hp*np.ones(len_eq_train)*length_ratio,
                                  (1-best_hp)*np.ones(len(X_train) - len_eq_train)))
      else:
        hp[tuning_hp_name] = best_hp
      print('Successfully tuned hyperparameters.')

    # Define the model
    nnetwork = keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun'],
                                        kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))
    nnetwork.add(keras.layers.Dense(2, activation='linear',
                                    kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))
    # Compile and fit the model
    nnetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=custom_mae)
    train_nn_start = time.time()
    history = nnetwork.fit(X_train, y_train, epochs = hp['n_epochs'], validation_data = (X_val, y_val),
                            batch_size = hp['batch_size'], sample_weight = w_train)
    
    # Plot the MSE history of the training
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Custom MSE')
    plt.savefig(os.path.join('results','training_history.png'))

    train_nn_end = time.time()
    train_nn_time = (train_nn_end - train_nn_start)/60
    print('NN training time: {:.3g} minutes.'.format(train_nn_time))

    df = pd.DataFrame({'loss':history.history['loss'], 'val_loss':history.history['val_loss']})
    df.to_csv(os.path.join('results','training_history.csv'))
    nnetwork.save(os.path.join('data', 'nn_model.h5'))
    print('Successfully completed Neural Network training.')

    # Retrieve the loss name
    try:
      loss_name = nnetwork.loss.__name__
    except AttributeError:
      loss_name = nnetwork.loss.name

  # If training only one model, load the parameters from the other model
  if (mode=='rf' or mode=='nn'):
    with open(os.path.join('data','train_summary.pkl'), 'rb') as f:
        rf_summary, nn_summary = pickle.load(f)
      
  # Save the training summary
  if (mode=='rf' or mode=='all'):
    rf_summary = "".join(['\n\n***MODEL TRAINING***',
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

  with open(os.path.join('data','train_summary.pkl'), 'wb') as f:
      pickle.dump([rf_summary, nn_summary], f)

  return

def hp_tuning(train_val_data, add_train_vars, hp, loss, tuning_hp_name, tuning_hp_vals):

  # Unpack the data
  X_train, X_val, y_train, y_val = train_val_data
  w_train, len_eq_train = add_train_vars

  # Create empty lists to store the losses
  losses = []

  # Make a mask to subset of the data for tuning, conserving the eq/jp ratio
  tuning_factor = 0.1
  tuning_mask, len_eq_tuning, len_jp_tuning = \
    subset_mask_stratified(tuning_factor, len(X_train), len_eq_train)

  # Subset the data for tuning
  X_tuning = X_train[tuning_mask]
  y_tuning = y_train[tuning_mask]
  if w_train is not None:
    w_tuning = w_train[tuning_mask]

  # Test each hyperparameter value
  for i, value in enumerate(tuning_hp_vals):
    print(f'Testing hp {i+1} of {len(tuning_hp_vals)}...')

    # Change the hyperparameter to the tuning value
    if tuning_hp_name == 'w_eq':
      # Generate the weights for the equilibrium and jumps data
      length_ratio = len_jp_tuning/len_eq_tuning
      w_tuning = np.concatenate((value*np.ones(len_eq_tuning)*length_ratio,
                                (1-value)*np.ones(len_jp_tuning)))
    else:
      hp[tuning_hp_name] = value

    # Define the model
    nnetwork = keras.Sequential()
    for n_units in hp['units']:
      nnetwork.add(tf.keras.layers.Dense(units=n_units, activation=hp['act_fun'],
                                        kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))
    nnetwork.add(keras.layers.Dense(2, activation='linear',
                                    kernel_regularizer=keras.regularizers.l1(hp['l1_reg'])))

    # Compile and fit the model
    nnetwork.compile(optimizer=keras.optimizers.Adam(learning_rate=hp['learning_rate']), loss=loss)
    history = nnetwork.fit(X_tuning, y_tuning, epochs = hp['n_epochs'], validation_data = (X_val, y_val), 
                          batch_size = hp['batch_size'], sample_weight = w_tuning)
    losses.append(history.history['val_loss'][-1])

  # Save a csv with each model's loss and hp
  pd.DataFrame({'loss':losses, tuning_hp_name:tuning_hp_vals}).to_csv(os.path.join('results','hp_tuning.csv'))

  # Select the best hyperparameter and retrain the model
  best_hp = tuning_hp_vals[np.argmin(losses)]
  print('Best model has {} = {:.1g}.'.format(tuning_hp_name, best_hp))
  return best_hp