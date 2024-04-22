from dataclasses import dataclass
from collections import namedtuple
from pathlib import Path

@dataclass(frozen=True)
class main:
    model_training: str = 'nn'    # 'none', 'rf', 'nn' or 'all'.
    model_evaluation: str = 'all'  # 'none', 'train', 'test' or 'all'
    fwd_sim: tuple = ()   # Folder names in fwd_sim, e.g. ('train_sim', 'test_sim')
    plots: tuple = ('surface', 'colormesh')   # ['surface', 'colormesh', 'tipping']

@dataclass(frozen=True)
class data_preparation:

    # DATA SOURCE
    data_source: str = 'minimal'    # 'detailed' or 'minimal'

    # DATA LOADING (if data_source == 'detailed')
    data_folder: str = 'detailed_larger_jumps' # Name of the folder where the data is located
    load_all: bool = False            # Load all simulations
    first_sim: int = 1               # First simulation to load (ignored if load_all is True)
    last_sim: int = 2000             # Last simulation to load (ignored if load_all is True)

    # DATA GENERATION (if data_source == 'minimal')
    n_sim: int = 1000               # Number of simulations to generate
    n_years: int = 10000            # Number of years per simulation
    dt: float = 0.5                 # Time step in years
    prob_new_state: float = 0.02    # Probability of jumping to a new system state
    prob_new_g: float = 0.002       # Probability of jumping to a new g value

    # DATA FORMATTING
    mask_zeroes: bool = True        # Remove zeroes in the data (to speed up training)
    drop_size: float = 0.0          # Percentage of data to drop (to speed up training)
    test_size: float = 0.2          # Percentage of data to use for testing
    val_size: float = 0.1           # Percentage of data to use for validation

@dataclass(frozen=False)
class model_training:

    # DEFAULT NN HYPERPARAMETERS
    def get_nn_hp():
        return {
            'units': (9, 27, 81, 162, 324, 648, 1296),
            'act_fun': 'relu',
            'learning_rate': 1E-5,
            'batch_size': 2 ** 12,
            'l1_reg': 1e-5,
            'n_epochs': 200
        }

    # TUNING PARAMETERS
    tuning_size: float = 0.1        # Percentage of data to use for tuning
    tuning_hp_name: str = 'units'   # Name of the hyperparameter to tune (according to nn_hp)
    tuning_hp_vals: tuple = ()      # Values to test for the hyperparameter
        

@dataclass(frozen=True)
class forward_simulation:

    # SIMULATION PARAMETERS
    max_years = 10000   # Maximum number of years to simulate
    freq_progress = 0.2 # Frequency of progress updates


@dataclass(frozen=True)
class paths:

    # ROOT PATH
    root: Path = Path(__file__).resolve().parents[1]

    # DATA PATHS
    raw_data: Path = root / 'data' / 'raw'
    fwd_sim_data: Path = root / 'data' / 'raw' / 'fwd_sim'
    processed_data: Path = root / 'data' / 'processed'
    temp_data: Path = root / 'data' / 'temp'

    # RESULTS PATHS
    figures: Path = root / 'results' / 'figures'
    outputs: Path = root / 'results' / 'outputs'
    models: Path = root / 'results' / 'models'
