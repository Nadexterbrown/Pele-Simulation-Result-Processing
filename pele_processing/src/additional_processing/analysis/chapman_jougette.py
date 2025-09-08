"""
Chapman-Jouguet (CJ) Deflagration Solver

This module provides a robust solver for determining the Chapman-Jouguet deflagration
state from simulation gas data without requiring an initial velocity guess. The solver
systematically searches for the CJ condition where the burned gas reaches Mach 1.

Key Features:
- Automatic initial guess estimation based on gas properties
- Two-phase search algorithm (coarse then fine)
- Robust conservation equation solving
- Comprehensive error handling and validation
"""

import numpy as np
import cantera as ct
from scipy import optimize
from typing import Dict, Tuple, Optional, Union
import warnings


def soundspeed_fr(gas):
    """
    Calculate frozen sound speed for a gas mixture.
    
    Parameters:
    -----------
    gas : cantera.Solution
        Gas object with defined state
        
    Returns:
    --------
    float : Frozen sound speed (m/s)
    """
    return gas.sound_speed


def hug_eq(T, v, h0, P0, v0, gas):
    """
    Hugoniot equation for combustion products.
    
    Parameters:
    -----------
    T : float
        Temperature to solve for (K)
    v : float
        Specific volume (m�/kg)
    h0 : float
        Initial specific enthalpy (J/kg)
    P0 : float
        Initial pressure (Pa)
    v0 : float
        Initial specific volume (m�/kg)
    gas : cantera.Solution
        Gas object for products
        
    Returns:
    --------
    float : Hugoniot equation residual
    """
    try:
        gas.TD = T, 1.0/v
        gas.equilibrate('TV')
        h = gas.h
        P = gas.P
        return h - h0 - 0.5 * (P + P0) * (v - v0)
    except:
        return 1e10


class ChapmanJouguetSolver:
    """
    Robust Chapman-Jouguet deflagration solver that finds the CJ state
    without requiring an initial velocity guess.
    """
    
    def __init__(self, mechanism_file: str, tol_mach: float = 1e-4, 
                 tol_residual: float = 1e-8, max_iterations: int = 1000):
        """
        Initialize the CJ solver.
        
        Parameters:
        -----------
        mechanism_file : str
            Path to reaction mechanism file (YAML/XML)
        tol_mach : float
            Tolerance for M1 = 1.0 condition
        tol_residual : float
            Tolerance for conservation equations
        max_iterations : int
            Maximum iterations for iterative solver
        """
        self.mechanism = mechanism_file
        self.tol_mach = tol_mach
        self.tol_residual = tol_residual
        self.max_iterations = max_iterations
        
        # Initialize gas objects
        self.gas_initial = ct.Solution(mechanism_file)
        self.gas_products = ct.Solution(mechanism_file)
        self.gas_equilibrium = ct.Solution(mechanism_file)
    
    def generate_initial_guess(self, M0: float, T0: float, P0: float, 
                              composition: Union[str, Dict]) -> Tuple[float, float]:
        """
        Generate intelligent initial guess for M1 and rho1 based on upstream conditions.
        
        Parameters:
        -----------
        M0 : float
            Upstream Mach number
        T0, P0 : float
            Initial temperature (K) and pressure (Pa)
        composition : str or dict
            Gas composition
            
        Returns:
        --------
        tuple : (M1_guess, rho1_guess)
        """
        # Set initial gas state
        self.gas_initial.TPX = T0, P0, composition
        rho0 = self.gas_initial.density
        
        # Calculate adiabatic flame temperature for density estimate
        self.gas_equilibrium.TPX = T0, P0, composition
        self.gas_equilibrium.equilibrate('HP')
        T_adiabatic = self.gas_equilibrium.T
        
        # Empirical relations based on typical deflagration behavior
        # M1 decreases as M0 increases, approaching 1.0 at CJ condition
        M1_guess = 0.95 - 0.3 * M0**0.5
        M1_guess = max(0.1, min(0.99, M1_guess))
        
        # Density increases due to combustion and compression
        # Account for thermal expansion and compression effects
        expansion_factor = T0 / T_adiabatic
        compression_factor = 1.0 + 2.0 * M0  # Empirical relation
        rho1_guess = rho0 * expansion_factor * compression_factor
        
        return M1_guess, rho1_guess
    
    def solve_conservation_equations(self, M0: float, T0: float, P0: float,
                                   composition: Union[str, Dict],
                                   initial_guess: Optional[Tuple[float, float]] = None) -> Tuple[float, float, list, bool]:
        """
        Solve conservation equations for given upstream Mach number.
        
        Parameters:
        -----------
        M0 : float
            Upstream Mach number
        T0, P0 : float
            Initial temperature and pressure
        composition : str or dict
            Gas composition
        initial_guess : tuple, optional
            Initial guess for (M1, rho1)
            
        Returns:
        --------
        tuple : (M1, rho1, residuals, converged_flag)
        """
        # Set initial state
        self.gas_initial.TPX = T0, P0, composition
        rho0 = self.gas_initial.density
        a0 = soundspeed_fr(self.gas_initial)
        u0 = M0 * a0
        h0 = self.gas_initial.h
        
        # Generate initial guess if not provided
        if initial_guess is None:
            M1_guess, rho1_guess = self.generate_initial_guess(M0, T0, P0, composition)
        else:
            M1_guess, rho1_guess = initial_guess
        
        def conservation_residuals(variables):
            """Residual function for conservation equations."""
            M1, rho1 = variables
            
            # Physical constraints
            if M1 <= 0 or M1 > 1.2 or rho1 <= 0 or rho1 > 50*rho0:
                return [1e10, 1e10]
            
            try:
                # Solve for temperature using Hugoniot relation
                v1 = 1.0 / rho1
                v0 = 1.0 / rho0
                
                # Use robust solver for temperature
                T1_solutions = optimize.fsolve(
                    hug_eq, T0 * 2.0,  # Initial guess: twice initial temperature
                    args=(v1, h0, P0, v0, self.gas_products),
                    xtol=1e-8, maxfev=500
                )
                T1 = T1_solutions[0]
                
                # Validate temperature
                if T1 <= 0 or T1 > 5000:  # Reasonable temperature bounds
                    return [1e10, 1e10]
                
                # Set product state and equilibrate
                self.gas_products.TD = T1, rho1
                self.gas_products.X = composition
                self.gas_products.equilibrate('TV')
                
                P1 = self.gas_products.P
                a1 = soundspeed_fr(self.gas_products)
                u1 = M1 * a1
                
                # Conservation equations (normalized)
                mass_residual = (rho0 * u0 - rho1 * u1) / (rho0 * u0)
                momentum_residual = (P0 + rho0*u0**2 - P1 - rho1*u1**2) / (P0 + rho0*u0**2)
                
                return [mass_residual, momentum_residual]
                
            except Exception as e:
                return [1e10, 1e10]
        
        # Solve with multiple attempts for robustness
        converged = False
        attempts = 0
        max_attempts = 5
        
        while not converged and attempts < max_attempts:
            try:
                if attempts > 0:
                    # Perturb initial guess for subsequent attempts
                    perturbation = 0.1 * np.random.randn(2)
                    guess = [M1_guess * (1 + perturbation[0]), 
                            rho1_guess * (1 + perturbation[1])]
                else:
                    guess = [M1_guess, rho1_guess]
                
                # Ensure guess is within bounds
                guess[0] = max(0.01, min(1.1, guess[0]))
                guess[1] = max(0.1*rho0, min(20*rho0, guess[1]))
                
                solution = optimize.fsolve(
                    conservation_residuals, guess,
                    xtol=self.tol_residual, maxfev=self.max_iterations
                )
                
                # Verify solution quality
                residuals = conservation_residuals(solution)
                if all(abs(r) < 1e-6 for r in residuals):
                    converged = True
                    M1, rho1 = solution
                    break
                    
            except Exception as e:
                pass
            
            attempts += 1
        
        if not converged:
            return 0, 0, [1e10, 1e10], False
        
        return M1, rho1, residuals, converged
    
    def find_cj_bracket(self, T0: float, P0: float, 
                       composition: Union[str, Dict]) -> Tuple[Optional[float], Optional[float]]:
        """
        Phase 1: Find bracket containing the CJ point using coarse search.
        
        Parameters:
        -----------
        T0, P0 : float
            Initial temperature and pressure
        composition : str or dict
            Gas composition
            
        Returns:
        --------
        tuple : (M0_low, M0_high) bracketing the CJ point, or (None, None) if not found
        """
        M0_start = 0.001
        M0_max = 0.6
        initial_step = 0.02
        
        M0_current = M0_start
        step = initial_step
        previous_M1 = 0
        
        search_results = []
        
        while M0_current < M0_max:
            M1, rho1, residuals, converged = self.solve_conservation_equations(
                M0_current, T0, P0, composition
            )
            
            if converged:
                search_results.append({
                    'M0': M0_current,
                    'M1': M1,
                    'rho1': rho1,
                    'residuals': residuals
                })
                
                # Check for CJ condition approach
                if M1 > 0.95:
                    # Found potential CJ region
                    if len(search_results) > 1 and previous_M1 < 0.95:
                        # Bracket found: CJ point between previous and current M0
                        return search_results[-2]['M0'], M0_current
                    elif M1 > 1.05:
                        # Overshot - narrow search
                        return max(M0_current - 2*step, M0_start), M0_current
                
                # Adaptive step sizing
                if M1 > 0.9:
                    step = 0.002  # Fine steps near CJ
                elif M1 > 0.8:
                    step = 0.005
                elif M1 > 0.6:
                    step = 0.01
                else:
                    step = initial_step
                
                previous_M1 = M1
            
            M0_current += step
        
        # If we reached here without finding bracket, check last valid points
        if search_results:
            last_result = search_results[-1]
            if last_result['M1'] > 0.85:
                # Close to CJ, return narrow bracket around last point
                return max(last_result['M0'] - 0.02, M0_start), min(last_result['M0'] + 0.02, M0_max)
        
        return None, None
    
    def refine_cj_solution(self, M0_low: float, M0_high: float,
                          T0: float, P0: float, 
                          composition: Union[str, Dict]) -> Tuple[float, Tuple]:
        """
        Phase 2: Refine CJ solution using bisection/secant method.
        
        Parameters:
        -----------
        M0_low, M0_high : float
            Bracket for upstream Mach number
        T0, P0 : float
            Initial temperature and pressure
        composition : str or dict
            Gas composition
            
        Returns:
        --------
        tuple : (M0_cj, cj_solution_data)
        """
        max_refinement_iterations = 50
        iteration = 0
        
        while (M0_high - M0_low) > 1e-7 and iteration < max_refinement_iterations:
            M0_mid = (M0_low + M0_high) / 2.0
            
            M1, rho1, residuals, converged = self.solve_conservation_equations(
                M0_mid, T0, P0, composition
            )
            
            if not converged:
                # If convergence fails, try to adjust bracket
                M0_mid = 0.7 * M0_low + 0.3 * M0_high  # Bias toward lower M0
                M1, rho1, residuals, converged = self.solve_conservation_equations(
                    M0_mid, T0, P0, composition
                )
            
            if converged:
                if abs(M1 - 1.0) < self.tol_mach:
                    # Found CJ point
                    return M0_mid, (M1, rho1, residuals, converged)
                elif M1 < 1.0:
                    M0_low = M0_mid
                else:
                    M0_high = M0_mid
            else:
                # Convergence failed, try different approach
                M0_high = M0_mid
            
            iteration += 1
        
        # Return best estimate
        M0_final = (M0_low + M0_high) / 2.0
        final_solution = self.solve_conservation_equations(M0_final, T0, P0, composition)
        
        return M0_final, final_solution
    
    def find_cj_deflagration_state(self, T0: float, P0: float,
                                  composition: Union[str, Dict]) -> Dict:
        """
        Main function to find CJ deflagration state without initial velocity guess.
        
        Parameters:
        -----------
        T0 : float
            Initial temperature (K)
        P0 : float
            Initial pressure (Pa)
        composition : str or dict
            Gas composition (e.g., 'H2:2, O2:1' or {'H2': 2, 'O2': 1})
            
        Returns:
        --------
        dict : Complete CJ state information including:
            - cj_velocity: CJ propagation velocity (m/s)
            - M0, M1: Upstream and downstream Mach numbers
            - T0, T1: Upstream and downstream temperatures (K)
            - P0, P1: Upstream and downstream pressures (Pa)
            - rho0, rho1: Upstream and downstream densities (kg/m�)
            - composition_burned: Burned gas composition
            - converged: Boolean indicating successful convergence
        """
        try:
            # Initialize gas properties
            self.gas_initial.TPX = T0, P0, composition
            rho0 = self.gas_initial.density
            a0 = soundspeed_fr(self.gas_initial)
            h0 = self.gas_initial.h
            
            # Phase 1: Find bracket containing CJ point
            print(f"Phase 1: Searching for CJ bracket...")
            M0_bracket_low, M0_bracket_high = self.find_cj_bracket(T0, P0, composition)
            
            if M0_bracket_low is None:
                raise ValueError("Could not find CJ deflagration state bracket")
            
            print(f"Found CJ bracket: M0  [{M0_bracket_low:.6f}, {M0_bracket_high:.6f}]")
            
            # Phase 2: Refine to exact CJ point
            print(f"Phase 2: Refining CJ solution...")
            M0_cj, cj_solution = self.refine_cj_solution(
                M0_bracket_low, M0_bracket_high, T0, P0, composition
            )
            
            M1_cj, rho1_cj, residuals, converged = cj_solution
            
            if not converged:
                warnings.warn("CJ solution may not be fully converged")
            
            # Calculate CJ velocity
            u_cj = M0_cj * a0
            
            # Get complete burned gas state
            self.gas_products.TPX = T0, P0, composition
            v1 = 1.0 / rho1_cj
            v0 = 1.0 / rho0
            
            # Solve for burned gas temperature
            T1_solutions = optimize.fsolve(
                hug_eq, T0 * 2.5,
                args=(v1, h0, P0, v0, self.gas_products),
                xtol=1e-8
            )
            T1_cj = T1_solutions[0]
            
            # Set final burned state
            self.gas_products.TD = T1_cj, rho1_cj
            self.gas_products.equilibrate('TV')
            
            # Compile comprehensive results
            cj_results = {
                'cj_velocity': u_cj,
                'flame_speed': u_cj,  # Alternative name
                'M0': M0_cj,
                'M1': M1_cj,
                'T0': T0,
                'T1': self.gas_products.T,
                'P0': P0,
                'P1': self.gas_products.P,
                'rho0': rho0,
                'rho1': rho1_cj,
                'u0': M0_cj * a0,
                'u1': M1_cj * soundspeed_fr(self.gas_products),
                'a0': a0,
                'a1': soundspeed_fr(self.gas_products),
                'composition_initial': dict(zip(self.gas_initial.species_names, self.gas_initial.X)),
                'composition_burned': dict(zip(self.gas_products.species_names, self.gas_products.X)),
                'residuals': residuals,
                'converged': converged and abs(M1_cj - 1.0) < self.tol_mach,
                'convergence_error': abs(M1_cj - 1.0)
            }
            
            print(f"CJ Solution Found:")
            print(f"  CJ Velocity: {u_cj:.2f} m/s")
            print(f"  M0: {M0_cj:.6f}, M1: {M1_cj:.6f}")
            print(f"  T1/T0: {cj_results['T1']/T0:.3f}")
            print(f"  P1/P0: {cj_results['P1']/P0:.3f}")
            print(f"  �1/�0: {rho1_cj/rho0:.3f}")
            print(f"  Converged: {cj_results['converged']}")
            
            return cj_results
            
        except Exception as e:
            print(f"Error in CJ calculation: {str(e)}")
            return {
                'error': str(e),
                'converged': False,
                'cj_velocity': None
            }
    
    def validate_solution(self, cj_results: Dict) -> Dict:
        """
        Validate the CJ solution by checking physical constraints and conservation laws.
        
        Parameters:
        -----------
        cj_results : dict
            Results from find_cj_deflagration_state
            
        Returns:
        --------
        dict : Validation results with pass/fail status and detailed checks
        """
        if not cj_results.get('converged', False):
            return {'overall_valid': False, 'reason': 'Solution not converged'}
        
        validation = {'overall_valid': True, 'checks': {}}
        
        # Check 1: M1 H 1.0 (CJ condition)
        M1_error = abs(cj_results['M1'] - 1.0)
        validation['checks']['cj_condition'] = {
            'pass': M1_error < 1e-3,
            'value': cj_results['M1'],
            'error': M1_error
        }
        
        # Check 2: Physical property ranges
        T_ratio = cj_results['T1'] / cj_results['T0']
        P_ratio = cj_results['P1'] / cj_results['P0']
        rho_ratio = cj_results['rho1'] / cj_results['rho0']
        
        validation['checks']['temperature_ratio'] = {
            'pass': 1.5 < T_ratio < 10.0,
            'value': T_ratio
        }
        
        validation['checks']['pressure_ratio'] = {
            'pass': 1.0 < P_ratio < 50.0,
            'value': P_ratio
        }
        
        validation['checks']['density_ratio'] = {
            'pass': 1.0 < rho_ratio < 10.0,
            'value': rho_ratio
        }
        
        # Check 3: Conservation residuals
        max_residual = max(abs(r) for r in cj_results['residuals'])
        validation['checks']['conservation_residuals'] = {
            'pass': max_residual < 1e-5,
            'max_residual': max_residual
        }
        
        # Update overall validation
        validation['overall_valid'] = all(
            check['pass'] for check in validation['checks'].values()
        )
        
        return validation


def example_usage():
    """
    Example usage of the Chapman-Jouguet solver.
    """
    # Initialize solver (requires Cantera mechanism file)
    mechanism = 'h2o2.yaml'  # Example mechanism
    solver = ChapmanJouguetSolver(mechanism)
    
    # Define initial conditions
    T0 = 300.0  # K
    P0 = 101325.0  # Pa
    composition = 'H2:2, O2:1'  # Stoichiometric hydrogen-oxygen
    
    # Find CJ state
    cj_results = solver.find_cj_deflagration_state(T0, P0, composition)
    
    # Validate solution
    validation = solver.validate_solution(cj_results)
    
    print(f"\nValidation Results:")
    print(f"Overall Valid: {validation['overall_valid']}")
    for check_name, check_data in validation['checks'].items():
        print(f"  {check_name}: {'PASS' if check_data['pass'] else 'FAIL'}")
    
    return cj_results, validation


if __name__ == "__main__":
    # Run example
    results, validation = example_usage()