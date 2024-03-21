import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from config import forward_simulation as cfg
from config import paths

def forward_simulation(nnetwork, rforest, sim_name):

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

  axs[0].plot(t, D_true, '-b', label = 'Target values')
  axs[0].plot(t, D_nn, '-r', label = 'Neural network')
  axs[0].plot(t, D_for, '-g', label = 'Random forest')
  axs[0].set_ylim(0)
  axs[0].set_ylabel('soil thickness')
  axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator(2))
  axs[0].tick_params(axis="both", which="both", direction="in", 
                         top=True, right=True)

  axs[1].plot(t, B_true, '-b')
  axs[1].plot(t, B_nn, '-r')
  axs[1].plot(t, B_for, '-g')
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
  save_vars = {'t': t, 'B_true': B_true, 'D_true': D_true, 'g': g,
               'B_for': B_for, 'D_for': D_for, 'B_nn': B_nn, 'D_nn': D_nn}
  file_path = paths.outputs / f'fwd_sim_{sim_name}.csv'

  # Convert to 2D array and names list for saving
  save_vars_arr = np.column_stack(list(save_vars.values()))
  save_vars_names = ','.join(save_vars.keys())

  # Save 2D array to csv
  np.savetxt(file_path, save_vars_arr, delimiter=',', header=save_vars_names)

  # Compute the Pearson correlation coefficients
  r_for = np.corrcoef(B_true, B_for)[0, 1], np.corrcoef(D_true, D_for)[0, 1]
  r_nn = np.corrcoef(B_true, B_nn)[0, 1], np.corrcoef(D_true, D_nn)[0, 1]

  # Add a couple lines to the summary with the system evolution parameters
  evolution_summary = "".join(['\n\nSimulation {}:'.format(sim_name),
                               '\ntime_step = {}'.format(dt),
                               '\nn_years = {}'.format(n_years),
                               '\npearson_corr_for = {}'.format(r_for),
                               '\npearson_corr_nn = {}'.format(r_nn)])
  return evolution_summary

##############################################################################

def preprocess_fwd_sim_data(sim_name):
  
  # What folder to load the data from
  i = 1

  # Load the simulation data
  folder = paths.raw_data / cfg.fwd_data_folder / sim_name / str(i)
  biomass = np.loadtxt(folder / 'biomass.tss')[25:-1,1]
  soil_depth = np.loadtxt(folder / 'soildepth.tss')[25:-1,1]
  grazing_pressure = np.load(folder / 'grazing.npy')[25:-1] * 24 * 365

  # Load the jumps file if it exists
  try:
    jumps = np.loadtxt(folder / 'statevars_jumped.tss')[25:-1,1].astype(bool)
  except FileNotFoundError:
    jumps = np.zeros_like(biomass, dtype=bool)
    print(f'WARNING: No jumps file found for {sim_name} in folder {i}.')

  # Retrieve X from the data
  raw_X_sim = np.column_stack((biomass, soil_depth, grazing_pressure))

  # Pool the results by 26 steps
  X_sim = raw_X_sim.reshape(-1, 26, 3)
  X_sim = np.apply_along_axis(np.median, axis=1, arr=X_sim)
  jumps = jumps[::26]

  # Join into 2D array and save to csv
  sim_data = np.column_stack((biomass, soil_depth, grazing_pressure, jumps))
  np.savetxt(paths.processed_data / cfg.fwd_data_folder / f'{sim_name}.csv', sim_data,
             delimiter=',', header='B,D,g,jumps', comments='')
  
  return sim_data