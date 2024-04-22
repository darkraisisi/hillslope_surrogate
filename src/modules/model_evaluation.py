import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import joblib as jb
from keras.models import load_model
from config import paths
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def model_evaluation(mode='all'):

  # Load the models
  nnetwork = load_model(paths.models / 'nn_model.keras', compile=False)
  rforest = jb.load(paths.models / 'rf_model.joblib')

  # Initialize the summaries
  rf_eval_summary = ''
  nn_eval_summary = ''

  if mode == 'train' or mode == 'all':

    # Load the training data from csv files
    X_train = np.loadtxt(paths.processed_data / 'X_train.csv',
                         delimiter=",", skiprows=1)
    y_train = np.loadtxt(paths.processed_data / 'y_train.csv',
                         delimiter=",", skiprows=1)
    
    # Evaluate the training set
    print('Starting training set evaluation...')
    rf_train_summary, nn_train_summary = \
      evaluate_set(X_train, y_train, rforest, nnetwork, set_name='train')
    rf_eval_summary += rf_train_summary
    nn_eval_summary += nn_train_summary
    print('Successfully completed training set evaluation.')

  if mode == 'test' or mode == 'all':

    # Load the data from csv files
    X_test = np.loadtxt(paths.processed_data / 'X_test.csv',
                        delimiter=",", skiprows=1)
    y_test = np.loadtxt(paths.processed_data / 'y_test.csv',
                        delimiter=",", skiprows=1)
    
    # Evaluate the test set
    print('Starting test set evaluation...')
    rf_test_summary, nn_test_summary = \
      evaluate_set(X_test, y_test, rforest, nnetwork, set_name='test')
    rf_eval_summary += rf_test_summary
    nn_eval_summary += nn_test_summary
    print('Successfully completed test set evaluation.')

  return rf_eval_summary, nn_eval_summary

##############################################################################

def evaluate_set(X, y, rforest, nnetwork, set_name='train'):

  # Evaluate the MSE for the random forest
  y_pred_for = rforest.predict(X)
  mse_for = mean_squared_error(y, y_pred_for)
  print('MSE {} set (RF) = {:.5g}'.format(set_name, mse_for))

  rf_summary = '\nmse_{}= {}'.format(set_name, mse_for)

  # Define some variables for the plots
  min_B_test = min(np.min(y[:,0]), np.min(y_pred_for[:,0]))
  max_B_test = max(np.max(y[:,0]), np.max(y_pred_for[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y[:,1]), np.min(y_pred_for[:,1]))
  max_D_test = max(np.max(y[:,1]), np.max(y_pred_for[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Plot the predicted vs true values for dB/dt and dD/dt of RF
  fig, axs = plt.subplots(1, 2, figsize = (12,6))

  axs[0].plot(y[:,0], y[:,0], '-r')
  h0 = axs[0].hist2d(y[:,0], y_pred_for[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     norm = colors.LogNorm())
  axs[0].set_xlabel('True dB/dt values')
  axs[0].set_ylabel('Predicted dB/dt values')
  axs[0].autoscale()

  axs[1].plot(y[:,1], y[:,1], '-r')
  h1 = axs[1].hist2d(y[:,1], y_pred_for[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('True dD/dt values')
  axs[1].set_ylabel('Predicted dD/dt values')
  axs[1].autoscale()

  fig.colorbar(h0[3], ax=axs[0])
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Random forest')
  fig.patch.set_alpha(1)

  # Calculate R² value for RF for each plot and add it to the plot
  r2_for_0 = r2_score(y[:,0], y_pred_for[:,0])
  r2_for_1 = r2_score(y[:,1], y_pred_for[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_for_0:.2e}', transform=axs[0].transAxes, verticalalignment='top')
  axs[1].text(0.05, 0.95, f'R² = {r2_for_1:.2e}', transform=axs[1].transAxes, verticalalignment='top')
  plt.savefig(paths.figures / f'{set_name}_pred_vs_true_rf.png')

  # Predict the values from the test set and evaluate the MSE
  y_pred_nn = nnetwork.predict(X, verbose = False)
  mse_nn = mean_squared_error(y, y_pred_nn)
  print('MSE test set (NN) = {:.5g}'.format(mse_nn))

  nn_summary = '\nmse_{}= {}'.format(set_name, mse_nn)


  # Determine which should be the minimum and maximum width of the bins
  min_B_test = min(np.min(y[:,0]), np.min(y_pred_nn[:,0]))
  max_B_test = max(np.max(y[:,0]), np.max(y_pred_nn[:,0]))
  B_binwidth = (max_B_test - min_B_test)/50

  min_D_test = min(np.min(y[:,1]), np.min(y_pred_nn[:,1]))
  max_D_test = max(np.max(y[:,1]), np.max(y_pred_nn[:,1]))
  D_binwidth = (max_D_test - min_D_test)/50

  # Plot the predicted vs true values for dB/dt and dD/dt of NN
  fig, axs = plt.subplots(1, 2, figsize = (12,6))

  axs[0].plot(y[:,0], y[:,0], '-r')
  h0 = axs[0].hist2d(y[:,0], y_pred_nn[:,0], 
                     bins = np.arange(min_B_test, max_B_test + B_binwidth, B_binwidth), 
                     norm = colors.LogNorm())
  axs[0].set_xlabel('True dB/dt values')
  axs[0].set_ylabel('Predicted dB/dt values')
  axs[0].autoscale()

  axs[1].plot(y[:,1], y[:,1], '-r')
  h1 = axs[1].hist2d(y[:,1], y_pred_nn[:,1], 
                     bins = np.arange(min_D_test, max_D_test + D_binwidth, D_binwidth),
                     norm = colors.LogNorm())
  axs[1].set_xlabel('True dD/dt values')
  axs[1].set_ylabel('Predicted dD/dt values')
  axs[1].autoscale()

  fig.colorbar(h0[3], ax=axs[0])
  fig.colorbar(h1[3], ax=axs[1])
  fig.suptitle('Neural network')
  fig.patch.set_alpha(1)

  # Calculate R² value for NN for each plot and add it to the plot
  r2_nn_0 = r2_score(y[:,0], y_pred_nn[:,0])
  r2_nn_1 = r2_score(y[:,1], y_pred_nn[:,1])
  axs[0].text(0.05, 0.95, f'R² = {r2_nn_0:.2e}', transform=axs[0].transAxes, verticalalignment='top')
  axs[1].text(0.05, 0.95, f'R² = {r2_nn_1:.2e}', transform=axs[1].transAxes, verticalalignment='top')
  plt.savefig(paths.figures / 'test_predicted_vs_true_nn.png')

  # Save the results
  np.savez(paths.outputs / 'test_evaluation.npz', y=y, y_pred_for=y_pred_for, y_pred_nn=y_pred_nn)
  print('Successfully completed NN test set evaluation.')

  return rf_summary, nn_summary