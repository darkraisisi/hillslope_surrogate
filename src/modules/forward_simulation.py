import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import joblib as jb
from keras.models import load_model
from config import forward_simulation as cfg
from config import paths

def forward_simulation(sim_names):

  ev_summary = '\n\n*FORWARD SIMULATION*'

  # Load the models
  nnetwork = load_model(paths.models / 'nn_model.keras', compile=False)
  rforest = jb.load(paths.models / 'rf_model.joblib')

  for sim_idx, sim_name in enumerate(sim_names):

    print(f'Running simulation {sim_idx+1} of {len(sim_names)}...')

    # Load the simulation data
    sim_data = preprocess_fwd_sim_data(sim_name)
    B_true, D_true, g, jumps = sim_data.T

    # Define starting run parameters
    Bo = B_true[0]	  # initial value of B
    Do = D_true[0]   # initial value of D
    dt = 0.5 			  # time step in years
    n_years = min(cfg.max_years, dt*len(B_true)) # number of years to simulate
    n_steps = int(n_years/dt) # number of steps to simulate
    steps_progress = int(n_steps*cfg.freq_progress) # number of steps to show progress

    # Initialize B and D
    B_for = np.ones((n_steps)) * Bo
    D_for = np.ones((n_steps)) * Do

    B_nn = np.ones((n_steps)) * Bo
    D_nn = np.ones((n_steps)) * Do

    # Subset the ground truth variables to the number of steps
    B_true = B_true[:n_steps]
    D_true = D_true[:n_steps]
    g = g[:n_steps]
    jumps = jumps[:n_steps]

    # Allow the system to evolve
    for step in range(1,n_steps):

      # Show progress
      if step%steps_progress == 0:
        print(f'{sim_name}: {100*step/n_steps:.0f}% of steps completed...')
      
      # If the system has jumped, set the new values
      if jumps[step]:
        B_for[step] = B_true[step]
        D_for[step] = D_true[step]
        B_nn[step] = B_true[step]
        D_nn[step] = D_true[step]
        continue
      
      # Compute the derivatives
      nn_slopes = nnetwork.predict(np.array([[B_nn[step-1], D_nn[step-1], g[step-1]]]), verbose = False)
      for_slopes = rforest.predict(np.array([[B_for[step-1], D_for[step-1], g[step-1]]]))

      # Compute the new values, forced to be within physically possible results
      B_for[step] = np.maximum(B_for[step-1] + for_slopes.squeeze()[0]*dt, 0.0)
      D_for[step] = np.maximum(D_for[step-1] + for_slopes.squeeze()[1]*dt, 0.0)
      B_nn[step] = np.maximum(B_nn[step-1] + nn_slopes.squeeze()[0]*dt, 0.0)
      D_nn[step] = np.maximum(D_nn[step-1] + nn_slopes.squeeze()[1]*dt, 0.0)

    # Create the time vector
    t = np.linspace(0, n_years, n_steps)

    # Plot D(t), B(t) and g(t)
    fig, axs = plt.subplots(3, 1, figsize = (10,7.5))

    start = 0
    for i in range(len(jumps)):
      if jumps[i]:
        axs[0].plot(t[start:i], D_true[start:i], '-b')
        axs[0].plot(t[start:i], D_nn[start:i], '-r')
        axs[0].plot(t[start:i], D_for[start:i], '-g')
        axs[0].axvline(x=t[i], color='k')

        axs[1].plot(t[start:i], B_true[start:i], '-b')
        axs[1].plot(t[start:i], B_nn[start:i], '-r')
        axs[1].plot(t[start:i], B_for[start:i], '-g')
        axs[1].axvline(x=t[i], color='k')

        start = i

    # Plot the last segment
    axs[0].plot(t[start:], D_true[start:], '-b')
    axs[0].plot(t[start:], D_nn[start:], '-r')
    axs[0].plot(t[start:], D_for[start:], '-g')

    axs[1].plot(t[start:], B_true[start:], '-b')
    axs[1].plot(t[start:], B_nn[start:], '-r')
    axs[1].plot(t[start:], B_for[start:], '-g')

    axs[0].set_ylim(0)
    axs[0].set_ylabel('soil thickness')
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[0].tick_params(axis="both", which="both", direction="in", 
              top=True, right=True)

    axs[1].set_ylim(0)
    axs[1].set_ylabel('biomass')
    axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[1].tick_params(axis="both", which="both", direction="in", 
              top=True, right=True)

    axs[2].plot(t, g, '-b')
    axs[2].set_ylim(0)
    axs[2].set_ylabel('grazing pressure')
    axs[2].set_xlabel('time (years)')
    axs[2].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    axs[2].tick_params(axis="both", which="both", direction="in", 
              top=True, right=True)

    fig.patch.set_alpha(1)
    plt.setp(axs, xlim=(0, n_years))
    plt.savefig(paths.figures / f'fwd_sim_{sim_name}.png')

    # Save the results
    save_vars = {'t': t, 'jumps': jumps, 'g': g,
                'B_true': B_true, 'D_true': D_true,
                'B_for': B_for, 'D_for': D_for,
                'B_nn': B_nn, 'D_nn': D_nn}
    file_path = paths.outputs / f'fwd_sim_{sim_name}.csv'

    # Convert to 2D array and names list for saving
    save_vars_arr = np.column_stack(list(save_vars.values()))
    save_vars_names = ','.join(save_vars.keys())

    # Save 2D array to csv
    np.savetxt(file_path, save_vars_arr, delimiter=',', header=save_vars_names)

    # Compute the Pearson correlation coefficients
    r_for_B, r_nn_B = weighted_corr(jumps, B_true, B_for, B_nn)
    r_for_D, r_nn_D = weighted_corr(jumps, D_true, D_for, D_nn)

    # Add a couple lines to the summary with the system evolution parameters
    ev_summary += "".join(['\n\nSIMULATION \'{}\':'.format(sim_name),
                           '\ntime_step = {}'.format(dt),
                           '\nn_years = {}'.format(n_years),
                           '\npearson_corr_for = {}'.format((r_for_B, r_for_D)),
                           '\npearson_corr_nn = {}'.format((r_nn_B, r_nn_D))])
    
  print('Successfully ran all simulations.')

  return ev_summary

##############################################################################

def preprocess_fwd_sim_data(sim_name):
  # Load the simulation data
  folder = paths.fwd_sim_data / sim_name
  biomass = np.loadtxt(folder / 'biomass.tss')[25:-1,1]
  soil_depth = np.loadtxt(folder / 'soildepth.tss')[25:-1,1]
  grazing_pressure = np.load(folder / 'grazing.npy')[25:-1] * 24 * 365

  # Load the jumps file if it exists
  try:
    jumps = np.loadtxt(folder / 'statevars_jumped.tss')[25:-1,1].astype(bool)
  except FileNotFoundError:
    jumps = np.zeros_like(biomass, dtype=bool)
    print(f'WARNING: No jumps file found for {sim_name}.')

  # Retrieve X from the data
  raw_X_sim = np.column_stack((biomass, soil_depth, grazing_pressure))

  # Pool the results by 26 steps
  X_sim = raw_X_sim.reshape(-1, 26, 3)
  X_sim = np.apply_along_axis(np.median, axis=1, arr=X_sim)
  jumps = jumps[::26]

  # Join into 2D array and save to csv
  sim_data = np.column_stack((X_sim, jumps))
  np.savetxt(paths.processed_data / cfg.fwd_data_folder / f'{sim_name}.csv', sim_data,
             delimiter=',', header='B,D,g,jumps', comments='')
  
  return sim_data

##############################################################################

def weighted_corr(jumps, true_data, for_data, nn_data):
  r_for_segments = []
  r_nn_segments = []
  weights = []
  start = 0
  for i, jump in enumerate(jumps):
    if jump:
      segment_length = i - start
      if segment_length > 1:
        r_for_segments.append(np.corrcoef(true_data[start:i], for_data[start:i])[0, 1])
        r_nn_segments.append(np.corrcoef(true_data[start:i], nn_data[start:i])[0, 1])
        weights.append(segment_length)
      else:
        print(f"Warning: Ignored segment of length {segment_length} at index {start}.")
      start = i
  segment_length = len(true_data) - start
  if segment_length > 1:
    r_for_segments.append(np.corrcoef(true_data[start:], for_data[start:])[0, 1])
    r_nn_segments.append(np.corrcoef(true_data[start:], nn_data[start:])[0, 1])
    weights.append(segment_length)
  else:
    print(f"Warning: Ignored segment of length {segment_length} at the end of the data.")
  return np.average(r_for_segments, weights=weights), np.average(r_nn_segments, weights=weights)