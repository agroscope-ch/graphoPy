"""
SOPRA Model - Grapholita funebrana Utility Functions
Python translation of R functions for insect population dynamics modeling

This module contains all auxiliary functions needed for the SOPRA model
that simulates the development stages of Grapholita funebrana.

Original R code by Matthieu Wilhelm
Python translation: Generated from R source files
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional


# =============================================================================
# Basic Utility Functions (from 0Unit.R)
# =============================================================================

def rate(b1: float, b2: float, temp: float) -> float:
    """
    Calculate linearly dependent rate.
    
    Args:
        b1: Slope parameter
        b2: Intercept parameter  
        temp: Temperature
        
    Returns:
        Development rate (per hour)
    """
    return max((b1 * temp) - b2, 0.0) / 24.0


def get_trunk_temp(day: int, temp_air: float, solar_rad: float) -> Dict[str, float]:
    """
    Estimate trunk inner and outer temperature from air temperature and radiation.
    
    Args:
        day: Julian day of year
        temp_air: Air temperature
        solar_rad: Solar radiation
        
    Returns:
        Dictionary with temp_trunk and temp_trin
    """
    # Compute trunk temperature
    light_interception = 1.0 / (1.0 + math.exp(4.0 - (0.05 * (day - 50))))
    incident_radiation = max(solar_rad, 0.0) * (1.0 - light_interception)
    temp_trunk = temp_air + (0.02279 * incident_radiation)
    
    # Compute trunk inner temperature  
    temp_trin = temp_air + (0.0146 * incident_radiation)
    
    return {
        'temp_trunk': temp_trunk,
        'temp_trin': temp_trin
    }


# =============================================================================
# Constants and Initialization Functions (from Grapholita_funebrana_utils.R)
# =============================================================================

def assign_const_and_var_gfune() -> Dict[str, float]:
    """
    Define all model constants and parameters.
    
    Returns:
        Dictionary containing all model constants
    """
    return {
        # Number of substages in delay for different development stages
        'k_pup_w': 30,    # postdiapause development
        'k_pup_m': 30,    # postdiapause development  
        'k_adu_w': 10,    # female life span
        'k_adu_m': 10,    # male life span
        'k_egg1': 30,     # egg development in 1st generation
        'k_lar1': 30,     # larval development in 1st generation
        'k_pup1': 50,     # pupal development in 1st generation
        'k_adu1_w': 10,   # female development in 1st generation
        'k_adu1_m': 10,   # male development in 1st generation
        'k_egg2': 30,     # egg development in 2nd generation
        'k_lar2': 30,     # larval development in 2nd generation
        'k_max': 50,      # max length of arrays
        
        # Adult lifespans (degree days for reproduction)
        'lifespan_adu': 350,   # overwintering generation (Butturini 2002)
        'lifespan_adu1': 350,  # summer generation (Butturini 2002)
        
        # Linear rate parameters for postdiapause development
        'b1_pup_w': 0.0042030,  # females slope (ACW 2009)
        'b2_pup_w': 0.0494177,  # females neg. intercept (ACW 2009)
        'b1_pup_m': 0.0039659,  # males slope (ACW 2009) 
        'b2_pup_m': 0.0446596,  # males neg. intercept (ACW 2009)
        
        # Linear rate parameters for adult lifespan
        'b1_adu_w': 0.002913,   # females slope (Butturini 2002)
        'b2_adu_w': 0.016812,   # females neg. intercept (Butturini 2002)
        'b1_adu_m': 0.002913,   # males slope (Butturini 2002)
        'b2_adu_m': 0.016812,   # males neg. intercept (Butturini 2002)
        
        # Linear rate parameters for 1st generation development
        'b1_egg1': 0.012806,    # egg development slope (Butturini 2002)
        'b2_egg1': 0.109739,    # egg development neg. intercept (Butturini 2002)
        'b1_lar1': 0.005714,    # larval development slope (Charmillot 1979)
        'b2_lar1': 0.057143,    # larval development neg. intercept (Charmillot 1979)
        'b1_pup1': 0.009415,    # pupal development slope (Butturini 2002)
        'b2_pup1': 0.115097,    # pupal development neg. intercept (Butturini 2002)
        'b1_adu1_w': 0.002913,  # 1st gen females lifespan slope (Butturini 2002)
        'b2_adu1_w': 0.016812,  # 1st gen females lifespan neg. intercept (Butturini 2002)
        'b1_adu1_m': 0.002913,  # 1st gen males lifespan slope (Butturini 2002)
        'b2_adu1_m': 0.016812,  # 1st gen males lifespan neg. intercept (Butturini 2002)
        
        # Linear rate parameters for 2nd generation development
        'b1_egg2': 0.012806,    # egg development slope (Butturini 2002)
        'b2_egg2': 0.109739,    # egg development neg. intercept (Butturini 2002)
        'b1_lar2': 0.002459,    # larval development slope (Butturini 2002)
        'b2_lar2': 0.013186,    # larval development neg. intercept (Butturini 2002)
        
        # Time step
        'r_dt': 1  # 1 day or 24 hours
    }


def init_param_gfune() -> Dict[str, bool]:
    """
    Initialize model parameters (triggers and flags).
    
    Returns:
        Dictionary with initial parameter values
    """
    return {
        'r_start': True,
        'flight_w': False,
        'flight_m': False, 
        'oviposition': False,      # No eggs laid yet
        'hatching': False,         # No larvae hatched yet
        'pupation': False,
        'flight1_w': False,        # No adults of second generation emerged yet
        'flight1_m': False,
        'oviposition1': False,
        'hatching1': False
    }


def init_value_gfune() -> Dict[str, float]:
    """
    Initialize population values for all life stages.
    
    Returns:
        Dictionary with initial population values (all zeros)
    """
    return {
        'pupae_w': 0.0,
        'pupae_m': 0.0,
        'adults_w': 0.0,
        'adults_m': 0.0,
        'eggs1': 0.0,
        'larvae1': 0.0,
        'pupae1': 0.0,
        'adults1_w': 0.0,
        'adults1_m': 0.0,
        'adults1': 0.0,
        'eggs2': 0.0,
        'larvae2': 0.0,
        'diap': 0.0
    }


# =============================================================================
# Delay Attribution Functions (from del_attribution_fun.R)
# =============================================================================

def set_del(b1: float, b2: float, temp: float) -> float:
    """
    Calculate delay time from rate parameters and temperature.
    
    Args:
        b1: Slope parameter
        b2: Intercept parameter
        temp: Temperature
        
    Returns:
        Delay time (hours)
    """
    rate_val = rate(b1, b2, temp)
    if rate_val > 0.0:
        return 1.0 / rate_val
    else:
        return 3.4 * math.exp(87)  # Very large delay for zero rate


def eval_rate(stage: str, temp_type: str, constants: Dict[str, float], 
              temp_values: Dict[str, float]) -> float:
    """
    Evaluate development rate for a given stage and temperature type.
    
    Args:
        stage: Development stage (e.g., 'pup_w', 'egg1')
        temp_type: Temperature type ('trunk' or 'air')
        constants: Dictionary with b1_ and b2_ parameters
        temp_values: Dictionary with temperature values
        
    Returns:
        Delay time for the stage
    """
    b1_key = f'b1_{stage}'
    b2_key = f'b2_{stage}'
    temp_key = f'temp_{temp_type}'
    
    return set_del(
        b1=constants[b1_key],
        b2=constants[b2_key], 
        temp=temp_values[temp_key]
    )


def del_naming_fun(stage: str, temp_type: str, del_var_df: List[Dict[str, str]]) -> str:
    """
    Generate delay variable name based on stage and temperature type.
    
    Args:
        stage: Development stage
        temp_type: Temperature type
        del_var_df: List of stage-temperature combinations
        
    Returns:
        Variable name for the delay (e.g., 'del_pup_w_S', 'del_egg1')
    """
    # Count how many entries exist for this stage
    stage_count = sum(1 for entry in del_var_df if entry['stage'] == stage)
    
    if stage_count > 1:
        # Multiple temperature types for this stage
        if temp_type == 'trunk':
            return f'del_{stage}_S'  # South (trunk)
        else:
            return f'del_{stage}_N'  # North (air)
    else:
        # Single temperature type for this stage
        return f'del_{stage}'


# =============================================================================
# Delay Loop Functions (Python equivalent of del_loop_fun_cpp)
# =============================================================================

def del_loop_fun(x_test: np.ndarray, a: float, vin: float, 
                 idt: int, ind_max: int) -> Tuple[np.ndarray, float]:
    """
    Python implementation of the delay loop function (equivalent to del_loop_fun_cpp).
    
    Args:
        x_test: State vector for substages
        a: Flow rate parameter
        vin: Input flow
        idt: Number of iterations
        ind_max: Maximum index (number of substages)
        
    Returns:
        Tuple of (updated state vector, output flow)
    """
    # Clone to avoid modifying original input
    x = x_test.copy()
    vout = 0.0
    
    for delloop in range(1, idt + 1):
        # Outflow from last compartment
        vout += a * x[ind_max - 1]
        
        # Update compartments (backwards from last to second)
        for i in range(ind_max - 1, 0, -1):
            x[i] = x[i] + a * x[i - 1] - a * x[i]
        
        # Inflow into first compartment (only on first iteration)
        inflow = a * vin * idt if delloop == 1 else 0.0
        
        # Update first compartment
        x[0] = x[0] + inflow - a * x[0]
    
    return x, vout


def del_loop_fun_adu(delrate: np.ndarray, a: float, vin: float, 
                     idt: int, ind_max: int, lifespan: float) -> Tuple[np.ndarray, float, float]:
    """
    Python implementation of adult delay loop with fecundity calculation.
    
    Args:
        delrate: State vector for adult substages
        a: Flow rate parameter
        vin: Input flow
        idt: Number of iterations
        ind_max: Number of substages
        lifespan: Adult lifespan for fecundity calculation
        
    Returns:
        Tuple of (updated state vector, output flow, cumulative fecundity)
    """
    # Clone to avoid modifying input
    x = delrate.copy()
    cumfec = 0.0
    vout = 0.0
    
    for delloop in range(1, idt + 1):
        # Outflow from last compartment
        vout += a * x[ind_max - 1]
        
        fec_act = 1.0
        
        # Update compartments and calculate fecundity (backwards)
        if ind_max >= 2:
            for i in range(ind_max - 1, 0, -1):
                # Update compartment i
                x[i] = x[i] + a * x[i - 1] - a * x[i]
                
                # Fecundity calculation
                fec_prev = 1.0 - math.exp(-0.0003323 * 
                                         ((i * (lifespan / ind_max) - 20.0) ** 1.726131))
                repro_rate = fec_act - fec_prev
                
                # Cumulative fecundity
                cumfec += repro_rate * (a * x[i])
                
                fec_act = fec_prev
        
        # Inflow into first compartment (only on first iteration)
        inflow = a * vin * idt if delloop == 1 else 0.0
        
        # Update first compartment
        fo1 = a * x[0]
        x[0] = x[0] + inflow - fo1
        
        # Fecundity contribution from first compartment
        cumfec += fec_act * fo1
    
    return x, vout, cumfec


# =============================================================================
# Block Delay Stage Functions
# =============================================================================

def block_delay_stage(active: bool, vin: float, del_val: float, k: int, 
                      r_dt: float, delrate: np.ndarray, 
                      trigger_name: Optional[str] = None) -> Dict[str, Union[np.ndarray, float, bool]]:
    """
    Block delay stage function for pupae, eggs, and larvae.
    
    Args:
        active: Whether this stage should run
        vin: Inflow into this stage
        del_val: Delay time
        k: Number of substages
        r_dt: Model timestep
        delrate: State vector of the stage
        trigger_name: Optional trigger name for next stage
        
    Returns:
        Dictionary with updated delrate, vout, and next_trigger
    """
    vout = 0.0
    next_trigger = False
    
    if active:
        delk = del_val / k
        idt = int(1.0 + (2.0 * (r_dt / delk)))
        a = (r_dt / delk) / idt
        
        if a > 0.99:
            a = 0.99
        
        # Call delay loop function
        delrate, vout = del_loop_fun(delrate, a, vin, idt, k)
        
        if vout > 0:
            next_trigger = True
    
    return {
        'delrate': delrate,
        'vout': vout,
        'next_trigger': next_trigger
    }


def block_delay_stage_adu(active: bool, vin: float, del_val: float, k: int,
                          r_dt: float, delrate: np.ndarray, lifespan: float,
                          trigger_name: Optional[str] = None) -> Dict[str, Union[np.ndarray, float, bool]]:
    """
    Block delay stage function for adults with fecundity calculation.
    
    Args:
        active: Whether this stage runs
        vin: Inflow into adult stage
        del_val: Delay time
        k: Number of substages
        r_dt: Timestep
        delrate: State vector for substages
        lifespan: Adult lifespan
        trigger_name: Optional trigger name
        
    Returns:
        Dictionary with updated delrate, vout, cumfec, and next_trigger
    """
    vout = 0.0
    cumfec = 0.0
    next_trigger = False
    
    if active:
        delk = del_val / k
        idt = int(1.0 + (2.0 * (r_dt / delk)))
        a = (r_dt / delk) / idt
        
        if a > 0.99:
            a = 0.99
        
        # Call adult delay loop function with fecundity
        delrate, vout, cumfec = del_loop_fun_adu(delrate, a, vin, idt, k, lifespan)
        
        if vout > 0:
            next_trigger = True
    
    return {
        'delrate': delrate,
        'vout': vout,
        'cumfec': cumfec,
        'next_trigger': next_trigger
    }


# =============================================================================
# Helper Functions
# =============================================================================

def create_del_var_df() -> List[Dict[str, str]]:
    """
    Create the delay variable dataframe equivalent.
    
    Returns:
        List of dictionaries with stage and temperature type combinations
    """
    stages = ['pup_w', 'pup_w', 'pup_m', 'pup_m', 'adu_w', 'adu_m', 
              'egg1', 'lar1', 'pup1', 'adu1_w', 'adu1_m', 'egg2', 'lar2']
    temps = ['trunk', 'air', 'trunk', 'air'] + ['air'] * 4 + ['trunk'] + ['air'] * 4
    
    return [{'stage': stage, 'temp': temp} for stage, temp in zip(stages, temps)]


def initialize_delrate_arrays(constants: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Initialize all delrate arrays with zeros.
    
    Args:
        constants: Dictionary with k_ values
        
    Returns:
        Dictionary with initialized delrate arrays
    """
    return {
        'delrate_pup_w_S': np.zeros(int(constants['k_pup_w'])),
        'delrate_pup_w_N': np.zeros(int(constants['k_pup_w'])),
        'delrate_pup_m_S': np.zeros(int(constants['k_pup_m'])),
        'delrate_pup_m_N': np.zeros(int(constants['k_pup_m'])),
        'delrate_adu_w': np.zeros(int(constants['k_adu_w'])),
        'delrate_adu_m': np.zeros(int(constants['k_adu_m'])),
        'delrate_egg1': np.zeros(int(constants['k_egg1'])),
        'delrate_lar1': np.zeros(int(constants['k_lar1'])),
        'delrate_pup1': np.zeros(int(constants['k_pup1'])),
        'delrate_adu1_w': np.zeros(int(constants['k_adu1_w'])),
        'delrate_adu1_m': np.zeros(int(constants['k_adu1_m'])),
        'delrate_egg2': np.zeros(int(constants['k_egg2'])),
        'delrate_lar2': np.zeros(int(constants['k_lar2']))
    }


# =============================================================================
# Main SOPRA Model Function
# =============================================================================

def update_gfune(values: Dict[str, float], day: int, hour: int, temp_air: float, 
                 solar_rad: float, temp_soil: float, 
                 curr_param: Optional[Dict] = None, 
                 constants: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
    """
    Main SOPRA model function for updating Grapholita funebrana population dynamics.
    
    This function simulates one time step of the insect population development
    through all life stages using temperature-dependent development rates and
    delayed response models.
    
    Args:
        values: Dictionary with current population values for all life stages
        day: Julian day of year (1-365)
        hour: Hour of day (0-23)
        temp_air: Air temperature (°C)
        solar_rad: Solar radiation
        temp_soil: Soil temperature (°C) - currently unused
        curr_param: Current model parameters (triggers, delrate arrays). If None, initializes for season start
        constants: Model constants. If None, uses default constants
        
    Returns:
        Dictionary with two keys:
        - 'updated_values': Updated population values for all life stages
        - 'current_param': Updated parameters for next time step
    """
    
    # =================================================================
    # INITIALIZATION AND SETUP
    # =================================================================
    
    # Calculate trunk temperatures from meteorological inputs
    trunk_temps = get_trunk_temp(day, temp_air, solar_rad)
    temp_trunk = trunk_temps['temp_trunk']
    temp_trin = trunk_temps['temp_trin']
    
    # Temperature values for delay calculations
    temp_values = {
        'temp_trunk': temp_trunk,
        'temp_air': temp_air,
        'temp_trin': temp_trin
    }
    
    # Initialize constants if not provided
    if constants is None:
        constants = assign_const_and_var_gfune()
    
    # Initialize current parameters if not provided (start of season)
    if curr_param is None:
        print("start of the season")
        curr_param = init_param_gfune()
        # Initialize delrate arrays
        delrate_arrays = initialize_delrate_arrays(constants)
        curr_param.update(delrate_arrays)
    
    # Extract parameters and values to local variables
    # Parameters (flags and triggers)
    r_start = curr_param['r_start']
    flight_w = curr_param.get('flight_w', False)
    flight_m = curr_param.get('flight_m', False)
    oviposition = curr_param.get('oviposition', False)
    hatching = curr_param.get('hatching', False)
    pupation = curr_param.get('pupation', False)
    flight1_w = curr_param.get('flight1_w', False)
    flight1_m = curr_param.get('flight1_m', False)
    oviposition1 = curr_param.get('oviposition1', False)
    hatching1 = curr_param.get('hatching1', False)
    
    # Population values
    pupae_w = values['pupae_w']
    pupae_m = values['pupae_m']
    adults_w = values['adults_w']
    adults_m = values['adults_m']
    eggs1 = values['eggs1']
    larvae1 = values['larvae1']
    pupae1 = values['pupae1']
    adults1_w = values['adults1_w']
    adults1_m = values['adults1_m']
    adults1 = values['adults1']
    eggs2 = values['eggs2']
    larvae2 = values['larvae2']
    diap = values['diap']
    
    # Delrate arrays
    delrate_pup_w_S = curr_param.get('delrate_pup_w_S', np.zeros(int(constants['k_pup_w'])))
    delrate_pup_w_N = curr_param.get('delrate_pup_w_N', np.zeros(int(constants['k_pup_w'])))
    delrate_pup_m_S = curr_param.get('delrate_pup_m_S', np.zeros(int(constants['k_pup_m'])))
    delrate_pup_m_N = curr_param.get('delrate_pup_m_N', np.zeros(int(constants['k_pup_m'])))
    delrate_adu_w = curr_param.get('delrate_adu_w', np.zeros(int(constants['k_adu_w'])))
    delrate_adu_m = curr_param.get('delrate_adu_m', np.zeros(int(constants['k_adu_m'])))
    delrate_egg1 = curr_param.get('delrate_egg1', np.zeros(int(constants['k_egg1'])))
    delrate_lar1 = curr_param.get('delrate_lar1', np.zeros(int(constants['k_lar1'])))
    delrate_pup1 = curr_param.get('delrate_pup1', np.zeros(int(constants['k_pup1'])))
    delrate_adu1_w = curr_param.get('delrate_adu1_w', np.zeros(int(constants['k_adu1_w'])))
    delrate_adu1_m = curr_param.get('delrate_adu1_m', np.zeros(int(constants['k_adu1_m'])))
    delrate_egg2 = curr_param.get('delrate_egg2', np.zeros(int(constants['k_egg2'])))
    delrate_lar2 = curr_param.get('delrate_lar2', np.zeros(int(constants['k_lar2'])))
    
    # =================================================================
    # DELAY RATE CALCULATIONS
    # =================================================================
    
    # Create delay variable dataframe equivalent
    del_var_df = create_del_var_df()
    
    # Calculate delay rates for all stage-temperature combinations
    delays = {}
    for entry in del_var_df:
        stage = entry['stage']
        temp_type = entry['temp']
        delay_name = del_naming_fun(stage, temp_type, del_var_df)
        delay_value = eval_rate(stage, temp_type, constants, temp_values)
        delays[delay_name] = delay_value
    
    # Extract individual delay values
    del_pup_w_S = delays['del_pup_w_S']
    del_pup_w_N = delays['del_pup_w_N']
    del_pup_m_S = delays['del_pup_m_S']
    del_pup_m_N = delays['del_pup_m_N']
    del_adu_w = delays['del_adu_w']
    del_adu_m = delays['del_adu_m']
    del_egg1 = delays['del_egg1']
    del_lar1 = delays['del_lar1']
    del_pup1 = delays['del_pup1']
    del_adu1_w = delays['del_adu1_w']
    del_adu1_m = delays['del_adu1_m']
    del_egg2 = delays['del_egg2']
    del_lar2 = delays['del_lar2']
    
    # =================================================================
    # INITIAL INFLOW CALCULATIONS (SEASON START)
    # =================================================================
    
    if r_start:
        # Initial inflow for overwintering pupae
        # 50% of female and male pupae hibernating in southern and 50% in northern trunk sector
        vin_pup_w_S = 0.8 * (del_pup_w_S / constants['k_pup_w'])
        vin_pup_w_N = 0.3 * (del_pup_w_N / constants['k_pup_w'])
        vin_pup_m_S = 0.8 * (del_pup_m_S / constants['k_pup_m'])
        vin_pup_m_N = 0.3 * (del_pup_m_N / constants['k_pup_m'])
        r_start = False
    else:
        vin_pup_w_S = 0.0
        vin_pup_w_N = 0.0
        vin_pup_m_S = 0.0
        vin_pup_m_N = 0.0
    
    # =================================================================
    # OVERWINTERING GENERATION - PUPAL DEVELOPMENT
    # =================================================================
    
    # Pupal females (South, overwintering generation)
    res = block_delay_stage(
        active=True,
        vin=vin_pup_w_S,
        del_val=del_pup_w_S,
        k=int(constants['k_pup_w']),
        r_dt=constants['r_dt'],
        delrate=delrate_pup_w_S,
        trigger_name="flight_w"
    )
    delrate_pup_w_S = res['delrate']
    vout_pup_w_S = res['vout']
    flight_w = res['next_trigger']
    
    # Pupal females (North, overwintering generation)
    res = block_delay_stage(
        active=True,
        vin=vin_pup_w_N,
        del_val=del_pup_w_N,
        k=int(constants['k_pup_w']),
        r_dt=constants['r_dt'],
        delrate=delrate_pup_w_N,
        trigger_name="flight_w"
    )
    delrate_pup_w_N = res['delrate']
    vout_pup_w_N = res['vout']
    flight_w = flight_w or res['next_trigger']  # Combine triggers
    
    # Calculate inflow to adult females
    if flight_w:
        vin_adu_w = (vout_pup_w_S * (del_adu_w / constants['k_adu_w']) + 
                     vout_pup_w_N * (del_adu_w / constants['k_adu_w']))
    else:
        vin_adu_w = 0.0
    
    # Pupal males (South, overwintering generation)
    res = block_delay_stage(
        active=True,
        vin=vin_pup_m_S,
        del_val=del_pup_m_S,
        k=int(constants['k_pup_m']),
        r_dt=constants['r_dt'],
        delrate=delrate_pup_m_S,
        trigger_name="flight_m"
    )
    delrate_pup_m_S = res['delrate']
    vout_pup_m_S = res['vout']
    flight_m = res['next_trigger']
    
    # Pupal males (North, overwintering generation)
    res = block_delay_stage(
        active=True,
        vin=vin_pup_m_N,
        del_val=del_pup_m_N,
        k=int(constants['k_pup_m']),
        r_dt=constants['r_dt'],
        delrate=delrate_pup_m_N,
        trigger_name="flight_m"
    )
    delrate_pup_m_N = res['delrate']
    vout_pup_m_N = res['vout']
    flight_m = flight_m or res['next_trigger']  # Combine triggers
    
    # Calculate inflow to adult males
    if flight_m:
        vin_adu_m = (vout_pup_m_S * (del_adu_m / constants['k_adu_m']) + 
                     vout_pup_m_N * (del_adu_m / constants['k_adu_m']))
    else:
        vin_adu_m = 0.0
    
    # =================================================================
    # OVERWINTERING GENERATION - ADULT DEVELOPMENT
    # =================================================================
    
    # Adult females (overwintering)
    res = block_delay_stage_adu(
        active=bool(flight_w),
        vin=vin_adu_w,
        del_val=del_adu_w,
        k=int(constants['k_adu_w']),
        r_dt=constants['r_dt'],
        delrate=delrate_adu_w,
        lifespan=constants['lifespan_adu']
    )
    delrate_adu_w = res['delrate']
    cumfec = res['cumfec']
    oviposition = res['next_trigger']
    
    # Calculate inflow to first generation eggs
    if flight_m:
        vin_egg1 = cumfec * (del_egg1 / constants['k_egg1'])
    else:
        vin_egg1 = 0.0
    
    # Adult males (overwintering)
    res = block_delay_stage(
        active=bool(flight_m),
        vin=vin_adu_m,
        del_val=del_adu_m,
        k=int(constants['k_adu_m']),
        r_dt=constants['r_dt'],
        delrate=delrate_adu_m,
        trigger_name="adu_m"
    )
    delrate_adu_m = res['delrate']
    # no vout needed here, since only lifespan is tracked
    
    # =================================================================
    # FIRST GENERATION DEVELOPMENT
    # =================================================================
    
    # Egg1 → Larvae (first generation)
    res = block_delay_stage(
        active=bool(oviposition),
        vin=vin_egg1,
        del_val=del_egg1,
        k=int(constants['k_egg1']),
        r_dt=constants['r_dt'],
        delrate=delrate_egg1,
        trigger_name="egg1"
    )
    delrate_egg1 = res['delrate']
    vout_egg1 = res['vout']
    hatching = res['next_trigger']
    
    # Calculate inflow to first generation larvae
    if hatching:
        vin_lar1 = vout_egg1 * (del_lar1 / constants['k_lar1'])
    else:
        vin_lar1 = 0.0
    
    # Larvae1 → Pupae1
    res = block_delay_stage(
        active=bool(hatching),
        vin=vin_lar1,
        del_val=del_lar1,
        k=int(constants['k_lar1']),
        r_dt=constants['r_dt'],
        delrate=delrate_lar1,
        trigger_name="lar1"
    )
    delrate_lar1 = res['delrate']
    vout_lar1 = res['vout']
    
    # Check conditions for pupation (before day 206)
    if (vout_lar1 > 0.0) and (day <= 206):
        pupation = True
        vin_pup1 = vout_lar1 * (del_pup1 / constants['k_pup1'])
    else:
        vin_pup1 = 0.0
    
    # Pupae1 → Adults (first summer generation)
    res = block_delay_stage(
        active=pupation,
        vin=vin_pup1,
        del_val=del_pup1,
        k=int(constants['k_pup1']),
        r_dt=constants['r_dt'],
        delrate=delrate_pup1,
        trigger_name="pup1"
    )
    delrate_pup1 = res['delrate']
    vout_pup1 = res['vout']
    flight1_w = res['next_trigger']
    flight1_m = res['next_trigger']
    
    # Calculate inflow to first generation adults
    vin_adu1_w = vout_pup1 * (del_adu1_w / constants['k_adu1_w'])
    vin_adu1_m = vout_pup1 * (del_adu1_m / constants['k_adu1_m'])
    
    # Adult females (first summer generation)
    res = block_delay_stage_adu(
        active=bool(flight1_w),
        vin=vin_adu1_w,
        del_val=del_adu1_w,
        k=int(constants['k_adu1_w']),
        r_dt=constants['r_dt'],
        delrate=delrate_adu1_w,
        lifespan=constants['lifespan_adu1']
    )
    delrate_adu1_w = res['delrate']
    cumfec1 = res['cumfec']
    oviposition1 = res['next_trigger']
    vin_egg2 = cumfec1 * (del_egg2 / constants['k_egg2'])
    
    # Adult males (first summer generation)
    res = block_delay_stage_adu(
        active=bool(flight1_m),
        vin=vin_adu1_m,
        del_val=del_adu1_m,
        k=int(constants['k_adu1_m']),
        r_dt=constants['r_dt'],
        delrate=delrate_adu1_m,
        lifespan=constants['lifespan_adu1']
    )
    delrate_adu1_m = res['delrate']
    
    # =================================================================
    # SECOND GENERATION DEVELOPMENT
    # =================================================================
    
    # Egg2 → Larvae2 (second generation)
    res = block_delay_stage(
        active=bool(oviposition1),
        vin=vin_egg2,
        del_val=del_egg2,
        k=int(constants['k_egg2']),
        r_dt=constants['r_dt'],
        delrate=delrate_egg2,
        trigger_name="egg2"
    )
    delrate_egg2 = res['delrate']
    vout_egg2 = res['vout']
    hatching1 = res['next_trigger']
    vin_lar2 = vout_egg2 * (del_lar2 / constants['k_lar2'])
    
    # Larvae2 (second generation)
    res = block_delay_stage(
        active=bool(hatching1),
        vin=vin_lar2,
        del_val=del_lar2,
        k=int(constants['k_lar2']),
        r_dt=constants['r_dt'],
        delrate=delrate_lar2,
        trigger_name="lar2"
    )
    delrate_lar2 = res['delrate']
    vout_lar2 = res['vout']
    
    # =================================================================
    # POPULATION UPDATES AND DIAPAUSE
    # =================================================================
    
    # Update population values by summing delrate arrays
    pupae_w = np.sum(delrate_pup_w_S) + np.sum(delrate_pup_w_N)
    pupae_m = np.sum(delrate_pup_m_S) + np.sum(delrate_pup_m_N)
    adults_w = np.sum(delrate_adu_w)
    adults_m = np.sum(delrate_adu_m)
    eggs1 = np.sum(delrate_egg1)
    larvae1 = np.sum(delrate_lar1)
    pupae1 = np.sum(delrate_pup1)
    adults1_w = np.sum(delrate_adu1_w)
    adults1_m = np.sum(delrate_adu1_m)
    eggs2 = np.sum(delrate_egg2)
    larvae2 = np.sum(delrate_lar2)
    adults1 = 0.5 * adults1_w + 0.5 * adults1_m
    
    # Diapause accumulation after day 212
    if day > 212:
        diap = diap + vout_lar1 + vout_lar2
    
    # =================================================================
    # RETURN RESULTS
    # =================================================================
    
    # Prepare updated values
    updated_values = {
        'pupae_w': pupae_w,
        'pupae_m': pupae_m,
        'adults_w': adults_w,
        'adults_m': adults_m,
        'eggs1': eggs1,
        'larvae1': larvae1,
        'pupae1': pupae1,
        'adults1_m': adults1_m,
        'adults1_w': adults1_w,
        'adults1': adults1,
        'eggs2': eggs2,
        'larvae2': larvae2,
        'diap': diap
    }
    
    # Prepare updated parameters
    current_param = {
        'r_start': r_start,
        'flight_w': flight_w,
        'flight_m': flight_m,
        'oviposition': oviposition,
        'hatching': hatching,
        'flight1_w': flight1_w,
        'flight1_m': flight1_m,
        'pupation': pupation,
        'oviposition1': oviposition1,
        'hatching1': hatching1,
        'delrate_pup_w_S': delrate_pup_w_S,
        'delrate_pup_w_N': delrate_pup_w_N,
        'delrate_pup_m_S': delrate_pup_m_S,
        'delrate_pup_m_N': delrate_pup_m_N,
        'delrate_adu_w': delrate_adu_w,
        'delrate_adu_m': delrate_adu_m,
        'delrate_egg1': delrate_egg1,
        'delrate_lar1': delrate_lar1,
        'delrate_pup1': delrate_pup1,
        'delrate_adu1_w': delrate_adu1_w,
        'delrate_adu1_m': delrate_adu1_m,
        'delrate_egg2': delrate_egg2,
        'delrate_lar2': delrate_lar2
    }
    
    return {
        'updated_values': updated_values,
        'current_param': current_param
    }