import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import matplotlib.ticker as tck
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc
import joblib as jb
from keras.models import load_model
from config import paths

def surface_plots(name='nn', g_plot = 1.76):

  # Load the model
  if name == 'nn':
    model = load_model(paths.models / 'nn_model.keras', compile=False)
  elif name == 'rf':
    model = jb.load(paths.models / 'rf_model.joblib')

  #Set the parameters
  scale_surface = 10
  n_sq = 18 * scale_surface
  B_lim = 3
  D_lim = 0.4

  # Make a grid of B and D values
  D_edges = np.linspace(0, D_lim, n_sq)
  B_edges = np.linspace(0, B_lim, n_sq)
  D_grid, B_grid = np.meshgrid(D_edges, B_edges)

  # Format the data to feed to the model
  g_grid =  np.ones((n_sq**2)) * g_plot
  X_grid = np.column_stack((B_grid.flatten(), D_grid.flatten(), g_grid))

  # Use the model to predict the value of the derivatives in the grid points
  Z = model.predict(X_grid)

  dB_dt = Z[:,0].reshape((n_sq,n_sq))
  dD_dt = Z[:,1].reshape((n_sq,n_sq))

  # Plot the surface for dB/dt and dD/dt for both models

  fig, ax = plt.subplots(1,2,figsize=(21,9), subplot_kw={"projection": "3d"})

  # Tweak the font size and resolution
  rcParams['font.size'] = 15
  rcParams['figure.dpi'] = 150

  # Format axis
  min_max_dB = [np.min(dB_dt), np.max(dB_dt)]
  ax[0].set_zticks(min_max_dB,min_max_dB)
  min_max_dD = [np.min(dD_dt), np.max(dD_dt)]
  ax[1].set_zticks(min_max_dD,min_max_dD)
  ax[0].get_proj = lambda: np.dot(Axes3D.get_proj(ax[0]), np.diag([1, 1, 0.3, 1]))
  ax[1].get_proj = lambda: np.dot(Axes3D.get_proj(ax[1]), np.diag([1, 1, 0.3, 1]))
  ax[0].zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
  ax[1].zaxis.set_major_formatter(FormatStrFormatter('%.4f'))
  ax[0].set_zlabel('Biomass net\ngrowth ($kg/m^2/yr$)', labelpad=34)
  ax[1].set_zlabel('Soil depth\nincrease (m/yr)', labelpad=36)

  for ax_ in ax:
    ax_.xaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))
    ax_.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_.tick_params(axis='z', pad=15)
    ax_.set_xlim(B_lim,0)
    ax_.set_ylim(0,D_lim)
    ax_.set_xlabel('Biomass ($kg/m^2$)', labelpad=20)
    ax_.set_ylabel('Soil depth ($m$)', labelpad=24)

  # Create a desaturated version of the colormap
  my_cmap = plt.cm.jet
  desaturation = 0.8
  jet_colors = my_cmap(np.arange(my_cmap.N))
  jet_colors_hsv = mc.rgb_to_hsv(jet_colors[:, :3])
  jet_colors_hsv[:, 1] *= desaturation
  jet_colors_desaturated = mc.hsv_to_rgb(jet_colors_hsv)
  my_cmap_desaturated = mc.ListedColormap(jet_colors_desaturated)

  # Plot the surface and eq. lines for dB/dt
  ax[0].plot_surface(B_grid, D_grid, dB_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
                    alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface)
  ax[0].set_zlim(np.min(dB_dt),np.max(dB_dt))
  dB_dt_0 = ax[0].contour3D(X=B_grid, Y=D_grid, Z=dB_dt, levels = [0.0], linewidths=0)

  grad_B, _ = np.gradient(dB_dt)
  st_eq_B, un_eq_B = eq_lines(dB_dt_0, grad_B, B_lim, D_lim, n_sq)

  for d_line in un_eq_B:
    ax[0].plot(d_line[:,0], d_line[:,1], zs=0, zdir='z', linestyle = 'dashed',
              linewidth=3, color = 'black', zorder=10)

  for s_line in st_eq_B:
    ax[0].plot(s_line[:,0], s_line[:,1], zs=0, zdir='z', linestyle = 'solid',
              linewidth=3, color = 'black', zorder=11)


  # Plot the surface and eq. lines for dD/dt
  ax[1].plot_surface(B_grid, D_grid, dD_dt, cmap=my_cmap_desaturated, linewidth=0.25, edgecolor = 'black',
                    alpha=1, shade=False, rstride=scale_surface, cstride=scale_surface)
  ax[1].set_zlim(np.min(dD_dt),np.max(dD_dt))
  dD_dt_0 = ax[1].contour3D(X=B_grid, Y=D_grid, Z=dD_dt, levels = [0.0], linewidths=0)

  _, grad_D = np.gradient(dD_dt)
  st_eq_D, un_eq_D = eq_lines(dD_dt_0, grad_D, B_lim, D_lim, n_sq)

  for d_line in un_eq_D:
    ax[1].plot(d_line[:,0], d_line[:,1], zs=0, zdir='z', linestyle = 'dashed',
              linewidth=4, color = 'black', zorder=10)

  for s_line in st_eq_D:
    ax[1].plot(s_line[:,0], s_line[:,1], zs=0, zdir='z', linestyle = 'solid',
              linewidth=4, color = 'black', zorder=11)

  plt.tight_layout()
  plt.savefig(paths.figures / f'surface_plot_{name}.png')
  #plt.savefig(os.path.join('results','surface_plot_nn.eps'), format='eps')

  # Plot the equilibrium lines
  fig, ax = plt.subplots(figsize=(16,14))
  ax.xaxis.set_major_locator(plt.MaxNLocator(3))
  ax.yaxis.set_major_locator(plt.MaxNLocator(3, prune='lower'))

  for solid_lines, dashed_lines, color, var in [[st_eq_B, un_eq_B, '#24A793', 'biomass'], [st_eq_D, un_eq_D, '#C00A35', 'soil depth']]:
    
    for i, d_line in enumerate(dashed_lines):
      d_line = np.array(d_line)
      ax.plot(d_line[:,1], d_line[:,0], linestyle = 'dashed', linewidth=5, color = color,
              label = f'Unstable {var} nullcline' if i==0 else "")

    for i, s_line in enumerate(solid_lines):
      s_line = np.array(s_line)
      ax.plot(s_line[:,1], s_line[:,0], linestyle = 'solid', linewidth=5, color = color,
              label = f'Stable {var} nullcline' if i==0 else "")

  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)')
  ax.set_xlabel('Soil depth ($m$)')
  ax.legend(loc = 'best', framealpha=1)
  plt.tight_layout()
  plt.savefig(paths.figures / f'eq_lines_{name}.png')
  #plt.savefig(os.path.join('results','eq_lines_nn.eps'), format='eps')

  # Find the feature space velocity and take the logarithm
  velocity = np.sqrt(dB_dt**2 + dD_dt**2)
  log_vel = np.log10(velocity)

  # Make the streamplot
  fig, ax = plt.subplots(figsize=(16,14))
  stream = plt.streamplot(D_grid, B_grid, dD_dt, dB_dt, color=log_vel, cmap=plt.cm.viridis,
                          minlength=0.01, linewidth=3, arrowsize=3)
  fig.colorbar(stream.lines)
  ax.set_ylim(0, B_lim)
  ax.set_xlim(0, D_lim)
  ax.set_ylabel('Biomass ($kg/m^2$)')
  ax.set_xlabel('Soil depth ($m$)')
  plt.tight_layout()
  plt.savefig(paths.figures / f'streamplot_{name}.png', facecolor='white')


  # Save the results
  print('Saving surface plot results...')
  df = pd.DataFrame({'B_grid':X_grid[:,0], 'D_grid':X_grid[:,1],
                     'dB_dt':Z[:,0], 'dD_dt':Z[:,1]})
  df.to_csv(paths.outputs / f'surface_plots_{name}.csv')
  print('Successfully saved surface plot results.')

  # Add a couple lines to the summary with the system evolution parameters
  surface_summary = "".join(['\n\n*SURFACE PLOTS*',
                             '\nn_sq = {}'.format(n_sq),
                             '\nB_lim = {}'.format(B_lim),
                             '\nD_lim = {}'.format(D_lim),
                             '\ng = {}'.format(g_plot)])

  return surface_summary

def eq_lines(contour, gradient, B_lim, D_lim, n_sq):
  # Find for what regions the equilibrium is stable
  grad_stab = gradient < 0

  dashed_lines = []
  solid_lines = [] 
  lines = contour.allsegs[0]
  for line in lines:
    if len(line) < 1:
      continue
    indices = (np.array(line)//np.array([(B_lim+1E-5)/n_sq, (D_lim+1E-6)/n_sq])).astype(int)
    stability = grad_stab[indices[:,0], indices[:,1]]
    current_line = line[0]

    for i in range(len(line)-1):
      if (stability[i] != stability[i+1]):
        midpoint = (line[i] + line[i+1])/2
        current_line = np.vstack([current_line, midpoint])
        if stability[i]:
          solid_lines.append(current_line)
        else:
          dashed_lines.append(current_line)
        current_line = midpoint
      current_line = np.vstack([current_line, line[i+1]])

    if stability[-1]:
          solid_lines.append(current_line)
    else:
      dashed_lines.append(current_line)
  return solid_lines, dashed_lines