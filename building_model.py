import math

from casadi import *  # Library for Optimization and Control
from scipy.linalg import expm, solve

##### Building Model #####

def ZHO_Integration(A_matrix, B_matrix, X, U, ts):
    """
    Perform a single integration step using the ZHO integration method.

    Parameters:
    - Ac (numpy.ndarray): The system matrix.
    - Bc (numpy.ndarray): The input matrix.
    - StepSize (float): The integration step size.
    - X (numpy.ndarray): The current state vector.
    - U (numpy.ndarray): The input vector.

    Returns:
    - numpy.ndarray: The new state vector after the integration step.
    """

    # Ensure arrays are numpy and 2D
    A_matrix = np.array(A_matrix, dtype=float)
    B_matrix = np.array(B_matrix, dtype=float)
    X = np.array(X, dtype=float).reshape(-1, 1)  # column vector
    U = np.array(U, dtype=float).ravel().reshape(-1, 1)  # column vector

    # Compute the matrix exponential of Ac multiplied by the step size
    A = expm(np.array(A_matrix) * ts)

    # Solve for B using the formula given
    B = np.linalg.solve(np.array(A_matrix), (A - np.eye(A_matrix.shape[0])) @ np.array(B_matrix))

    # Compute the next state
    X_new = A @ X + B @ U

    return X_new


def rc_building_model(timestep_minutes, non_hvac_kw_demand, hvac_kw_demand, indoor_temperature, outdoor_temperature):
    """
    One-State
    :param inputs:
        Cin: Thermal Capacitance of indoor air - constant - need from kunal
        Tam(t): Outdoor temperature - shape - have this already
        Tave(t): indoor temperature - shape - have
        Rwin: thermal resistance of windows - constant - need from kunal
        c1 scale for Qih
        c2 scale for Qac
        c3 scale for qsolar
        Qih: scaled heat gain due to internal heating - shape - need from kunal
        Qac: sclaed heat removal due to air conditioning - shape - need from kunal, also the controller will calculate this
        Qsolar: scaled heat gain due to solar radiation- - shape - need from kunal
        Qventi: heat gain / loss due to ventilation -
        Qinfil: Heat gain / loss due to infiltration
        x_k: last timestep temperature
        u_k: matrix of Control Components : [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]

        State Space RC Thermal Model
        A_matrix - system dynamics (heat balance).
        B_matrix - input dynamics (heat gains/losses).
        C_matrix - maps state to output temperature.
        x_k - previous state (indoor air temperature).
        u_k - control / input vector. [T_a, Q_sol1, Q_sol2, Q_Zic, Q_Zir, Q_ac]

    :return:
        tave(t) indoor temperature
    """

    # For simplicity, assuming heats are zero.
    q_int, q_venti, q_infil, q_sol, = 0,0,0,0

    parameters = {
        'num_states': 1,
        'num_inputs': 6,
        'r_win': 0.05,  # thermal resistance of windows K/kW
        'c_in': 1e6,  # thermal capacitance of indoor air (J/K) = 4.04 kWh/K
        'c1': 1.0,
        'c2': 0.967,
        'c3': 0.099
    }

    if math.isnan(indoor_temperature): indoor_temperature = 20.56  # 69 F Ideal Temperature
    timestep_seconds = timestep_minutes * 60

    q_ac = - hvac_kw_demand * timestep_minutes * 1000  # (J)

    f = timestep_minutes / 10  # original data timestep = 10

    num_states = parameters['num_states']
    num_inputs = parameters['num_inputs']
    r_win = parameters['r_win']
    c_in = parameters['c_in']
    c1 = parameters['c1']
    c2 = parameters['c2']
    c3 = parameters['c3']

    # System Matrix - 1-State System
    A_matrix = np.zeros((num_states, num_states))
    A_matrix[0, 0] = -1 / (r_win * c_in)

    # System Constants
    C_matrix = DM(1, num_states)
    C_matrix[:, :] = np.hstack(
        (np.array([[1]]), np.zeros((1, num_states - 1))))  # np.array([1,0]) for 2 state , y = cx, [1000] for 4 state

    B_matrix = np.zeros((num_states, num_inputs))
    B_matrix[0, 0] = 1 / (r_win * c_in)  # T_am (Outdoor Temperature)
    B_matrix[0, 1] = c1 / c_in  # Qih (Internal Heat)
    B_matrix[0, 2] = c2 / c_in  # Qac (Air Conditioning)
    B_matrix[0, 3] = 1 / c_in  # Qsolar
    B_matrix[0, 4] = 1 / c_in  # Qventi
    B_matrix[0, 5] = c3 / c_in  # Qinfil

    x_k = np.array([indoor_temperature])
    u_k = [outdoor_temperature, q_int * f, q_ac * f, q_sol * f, q_venti * f, q_infil * f]

    # USE ZH0 method
    # Use 60 Second Timestep
    x_k_1 = ZHO_Integration(A_matrix, B_matrix, x_k, u_k, timestep_seconds)  # Next Timestep Building Temperature

    building_kw_demand = hvac_kw_demand + non_hvac_kw_demand

    hvac_pf = 0.9
    non_hvac_pf = 0.98

    hvac_kvar_demand = hvac_kw_demand * np.tan(np.arccos(hvac_pf))
    non_hvac_kvar_demand = non_hvac_kw_demand * np.tan(np.arccos(non_hvac_pf))

    building_kvar_demand = hvac_kvar_demand + non_hvac_kvar_demand

    return x_k_1[0], building_kw_demand, building_kvar_demand