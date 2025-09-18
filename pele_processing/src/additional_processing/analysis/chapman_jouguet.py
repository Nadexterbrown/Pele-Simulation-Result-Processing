from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cantera as ct

from ..core.domain import CJProperties, CJType
from ..core.interfaces import ChapmanJouguetAnalyzer

class CJAnalyzer(ChapmanJouguetAnalyzer, CJType):

    def __init__(self, mech: Any, solver_type: str = "manual", verbose: bool = False):
        self.mech = mech
        self.solver_type = solver_type

    def analyze_cj_deflagration(self, T: float = 300, P: float = ct.one_atm, Y: Dict[str, float] = None) -> CJProperties:
        """Perform comprehensive CJ deflagration analysis.

        Args:
            T (float): Initial temperature in K.
            P (float): Initial pressure in Pa.
            Y (dict): Initial species mole fractions.

        Returns:
            CJProperties: Data structure containing CJ deflagration properties.
        """
        self.wave = CJType.DEFLAGRATION
        if self.solver_type == "manual":
            return self._cj_solver(T, P, Y)
        else:
            raise NotImplementedError(f"Solver type '{self.solver_type}' not implemented.")

    def analyze_cj_detonation(self, T: float = 300, P: float = ct.one_atm, Y: Dict[str, float] = None) -> CJProperties:
        """Perform comprehensive CJ detonation analysis.

        Args:
            T (float): Initial temperature in K.
            P (float): Initial pressure in Pa.
            Y (dict): Initial species mole fractions.

        Returns:
            CJProperties: Data structure containing CJ detonation properties.
        """

        def compare_methods(self) -> Dict:
            """
            Compare manual and SDToolbox implementations.
            """
            from sdtoolbox.postshock import CJspeed

            manual_result = self._cj_solver(T, P, Y)
            sdtoolbox_result = CJspeed(P, T, Y)

            return {
                'manual_velocity': manual_result.velocity,
                'sdtoolbox_velocity': sdtoolbox_result,
                'difference': abs(manual_result.velocity - sdtoolbox_result),
                'relative_error': abs(manual_result.velocity - sdtoolbox_result) / sdtoolbox_result
            }

        self.wave = CJType.DETONATION
        if self.solver_type == "manual":
            return self._cj_solver(T, P, Y)
        elif self.solver_type == "SDToolbox":
            try:
                from sdtoolbox.postshock import CJspeed
                return CJspeed(P, T, Y)
            except ImportError:
                raise ImportError("SDToolbox not installed. Please install it or use the 'manual' solver.")
        elif self.solver_type == "compare":
            return compare_methods(self)
        else:
            raise NotImplementedError(f"Solver type '{self.solver_type}' not implemented.")

    def _cj_solver(self, T: float, P: float, q: Any) -> CJProperties:
        """
        Unified Chapman-Jouguet solver for detonations and deflagrations.

        This solver implements the SDToolbox parabolic fitting method, which exploits
        the fact that the CJ point corresponds to an extremum (minimum for detonation,
        maximum for deflagration) in the velocity-density relationship.

        The physical principle: At the CJ point, the Rayleigh line is tangent to
        the Hugoniot curve, which mathematically manifests as an extremum in the
        shock velocity when plotted against density ratio.
        """

        """
        Initialize the CJ solver.

        Parameters:
        -----------
        T : float
            Initial temperature (K)
        P : float
            Initial pressure (Pa)
        q : str or dict
            Composition in Cantera format (e.g., 'H2:2, O2:1' or {'H2': 2, 'O2': 1})
        mech : str
            Path to mechanism file (e.g., 'gri30.yaml')
        wave : str
            Type of wave: 'detonation' or 'deflagration'
        verbose : bool
            If True, print convergence information
        """

        self.T1 = T
        self.P1 = P
        self.q = q
        self.mech = self.mech

        # Validate wave type
        if self.wave not in ['detonation', 'deflagration']:
            raise ValueError(f"Wave type must be 'detonation' or 'deflagration', got '{self.wave}'")

        # Initialize gas objects
        gas = ct.Solution(self.mech)  # Initial state
        # Set initial conditions
        gas.TPX = T, P, q

        # Solver parameters (from SDToolbox)
        self.ERRFT = 1.0e-4  # Temperature error tolerance
        self.ERRFV = 1.0e-4  # Velocity error tolerance
        self.numsteps = 20  # Number of points for parabolic fit

        if self.verbose:
            print(f"\nInitialized CJ Solver for {self.wave}")
            print(f"Initial conditions: T={T:.1f} K, P={P / 1e5:.3f} bar")
            print(f"Initial sound speed: {gas.sound_speed:.1f} m/s")

        # Solve for CJ velocity
        cj_speed, gas_final, R2 = self._cj_speed(T, P, q)

        # Calculate final state properties
        gas_final.equilibrate('TP')

        # Calculate Mach numbers
        mach_upstream = cj_speed / gas.sound_speed
        # At CJ point, flow is sonic relative to products
        mach_downstream = cj_speed * (gas.density_mass / gas_final.density_mass) / gas_final.sound_speed

        # Calculate additional properties
        gamma = gas_final.cp / gas_final.cv
        specific_energy = gas_final.int_energy_mass
        specific_volume = 1.0 / gas_final.density_mass
        density_ratio = gas_final.density_mass / gas.density_mass

        # Get species concentrations
        species_dict = {}
        for i, species in enumerate(gas_final.species_names):
            if gas_final.X[i] > 1e-10:  # Only include significant species
                species_dict[species] = gas_final.X[i]

        return CJProperties(
            cj_type=self.wave,
            pressure=gas_final.P,
            temperature=gas_final.T,
            density=gas_final.density,
            velocity=cj_speed,
            specific_energy=specific_energy,
            specific_volume=specific_volume,
            sound_speed=gas_final.sound_speed,
            gamma=gamma,
            mach_number=mach_upstream,
            product_mach_number=mach_upstream,
            density_ratio=density_ratio,
            enthalpy=gas_final.enthalpy_mass,
            r_squared=R2,
            converged=R2 > 0.999,
            species_concentrations=species_dict
        )


    def _cj_calc(self, gas, gas1, x):
        """
        Calculates the Chapman-Jouguet wave speed using Reynolds' iterative method.

        FUNCTION SYNTAX:
            [gas,w1] = CJ_calc(gas,gas1,ERRFT,ERRFV,x)

        INPUT:
            gas = working gas object
            gas1 = gas object at initial state
            ERRFT,ERRFV = error tolerances for iteration
            x = density ratio

        OUTPUT:
            gas = gas object at equilibrium state
            w1 = initial velocity to yield prescribed density ratio

        """
        T  = 2000
        r1 = gas1.density; V1 = 1/r1
        i = 0
        DT = 1000; DW = 1000

        # Initial guess
        V = V1 / x; r = 1 / V
        if self.wave == 'deflagration':
            w1 = 50  # Initial velocity guess for deflagration
        else:
            w1 = 2000

        # Get initial state
        [P, H] = self._eq_state(gas, r, T)

        # Start loop
        while (abs(DT) > self.ERRFT * T or abs(DW) > self.ERRFV * w1):
            i = i + 1
            if i == 500:
                'i = 500'
                return;
            # CALCULATE FH & FP FOR GUESS 1
            [FH, FP] = self._FHFP(w1, gas, gas1)
            # TEMPERATURE PERTURBATION
            DT = T * 0.02;
            Tper = T + DT; Vper = V; Rper = 1 / Vper; Wper = w1
            [Pper, Hper] = self._eq_state(gas, Rper, Tper)
            # CALCULATE FHX & FPX FOR "IO" STATE
            [FHX, FPX] = self._FHFP(Wper, gas, gas1)
            # ELEMENTS OF JACOBIAN
            DFHDT = (FHX - FH) / DT; DFPDT = (FPX - FP) / DT
            # VELOCITY PERTURBATION
            if self.wave == 'deflagration':
                # Use absolute perturbation for small velocities
                if w1 < 100:
                    DW = 1.0  # 1 m/s absolute perturbation
                else:
                    DW = 0.01 * w1  # Smaller relative perturbation
            else:
                DW = 0.02 * w1  # Original for detonation
            Wper = w1 + DW; Tper = T; Rper = 1 / V
            [Pper, Hper] = self._eq_state(gas, Rper, Tper)
            # CALCULATE FHX & FPX FOR "IO" STATE
            [FHX, FPX] = self._FHFP(Wper, gas, gas1)
            # ELEMENTS OF JACOBIAN
            DFHDW = (FHX - FH) / DW; DFPDW = (FPX - FP) / DW
            # INVERT MATRIX
            J = DFHDT * DFPDW - DFPDT * DFHDW
            b = [DFPDW, -DFHDW, -DFPDT, DFHDT]
            a = [-FH, -FP]
            DT = (b[0] * a[0] + b[1] * a[1]) / J
            DW = (b[2] * a[0] + b[3] * a[1]) / J
            # CHECK & LIMIT CHANGE VALUES
            # VOLUME
            DTM = 0.2 * T
            if abs(DT) > DTM:
                DT = DTM * DT / abs(DT)
            # Additional velocity change limiting for deflagration
            if self.wave == 'deflagration':
                if w1 < 10:
                    DWM = 5.0  # Absolute limit for very small velocities
                else:
                    DWM = 0.2 * w1  # Relative limit

                if abs(DW) > DWM:
                    DW = DWM * DW / abs(DW)

            # MAKE THE CHANGES
            T = T + DT
            w1 = w1 + DW

            # Enforce constraints for deflagration
            if self.wave == 'deflagration':
                # Keep velocity positive and subsonic
                if w1 <= 0.1:
                    w1 = 0.1
                elif w1 >= gas1.sound_speed:
                    w1 = 0.95 * gas1.sound_speed

            [P, H] = self._eq_state(gas, r, T)

        return gas, w1

    def _cj_speed(self, T: float, P: float, q: Any, fullOutput=False):
        """
        Calculates CJ detonation velocity for a given pressure, temperature, and
        composition.

        FUNCTION SYNTAX:
            If only CJ speed required:
            cj_speed = CJspeed(P1,T1,q,mech)
            If full output required:
            [cj_speed,R2,plot_data] = CJspeed(P1,T1,q,mech,fullOutput=True)

        INPUT:
            P1 = initial pressure (Pa)
            T1 = initial temperature (K)
            q = reactant species mole fractions in one of Cantera's recognized formats
            mech = cti file containing mechanism data (e.g. 'gri30.cti')

        OPTIONAL INPUT:
            fullOutput = set True for R-squared value and pre-formatted plot data
                        (the latter for use with sdtoolbox.utilities.CJspeed_plot)

        OUTPUT
            cj_speed = CJ detonation speed (m/s)
            R2 = R-squared value of LSQ curve fit (optional)
            plot_data = tuple (rr,w1,dnew,a,b,c)
                        rr = density ratio
                        w1 = speed
                        dnew = minimum density
                        a,b,c = quadratic fit coefficients

        """
        # DECLARATIONS
        if self.wave == 'deflagration':
            maxv = 0.8; minv = 0.05
        else:
            maxv = 2.0; minv = 1.5
        w1 = np.zeros(self.numsteps + 1, float)
        rr = np.zeros(self.numsteps + 1, float)

        gas1 = ct.Solution(self.mech)
        gas = ct.Solution(self.mech)
        # INTIAL CONDITIONS
        gas.TPX = T, P, q
        gas1.TPX = T, P, q
        # INITIALIZE ERROR VALUES & CHANGE VALUES
        i = 1
        T1 = gas1.T; P1 = gas1.P
        counter = 1;
        R2 = 0.0;
        cj_speed = 0.0
        a = 0.0;
        b = 0.0;
        c = 0.0;
        dnew = 0.0
        while (counter <= 4) or (R2 < 0.99999):
            step = (maxv - minv) / float(self.numsteps);
            i = 0;
            x = minv
            while x <= maxv:
                gas.TPX = T1, P1, q
                try:
                    [gas, temp] = self._cj_calc(gas, gas1, x)
                except:
                    i = i + 1;
                    x = x + step
                    continue
                w1[i] = temp
                rr[i] = gas.density / gas1.density
                i = i + 1;
                x = x + step
            [a, b, c, R2, SSE, SST] = self._lsq_cj_speed(rr, w1)
            dnew = -b / (2.0 * a)
            minv = dnew - dnew * 0.001
            maxv = dnew + dnew * 0.001
            counter = counter + 1
            cj_speed = a * dnew ** 2 + b * dnew + c

        # Get final gas state at CJ point
        gas.TPX = T, P, q
        final_gas, _ = self._cj_calc(gas, gas1, dnew)
        return cj_speed, final_gas, R2

    def _lsq_cj_speed(self, x, y):
        """
        Determines least squares fit of parabola to input data

        FUNCTION SYNTAX:
        [a,b,c,R2,SSE,SST] = LSQ_CJspeed(x,y)

        INPUT:
            x = independent data points
            y = dependent data points

        OUTPUT:
            a,b,c = coefficients of quadratic function (ax^2 + bx + c = 0)
            R2 = R-squared value
            SSE = sum of squares due to error
            SST = total sum of squares

        """
        # Calculate Sums
        k = 0
        X = 0.0;
        X2 = 0.0;
        X3 = 0.0;
        X4 = 0.0;
        Y = 0.0;
        Y1 = 0.0;
        Y2 = 0.0;
        a = 0.0;
        b = 0.0;
        c = 0.0;
        R2 = 0.0
        n = len(x)

        while k < n:
            X = X + x[k]
            X2 = X2 + x[k] ** 2
            X3 = X3 + x[k] ** 3
            X4 = X4 + x[k] ** 4
            Y = Y + y[k]
            Y1 = Y1 + y[k] * x[k]
            Y2 = Y2 + y[k] * x[k] ** 2
            k = k + 1
        m = float(Y) / float(n)

        den = (X3 * float(n) - X2 * X)
        temp = (den * (X * X2 - X3 * float(n)) + X2 * X2 * (X * X - float(n) * X2) - X4 * float(n) * (
                    X * X - X2 * float(n)))
        temp2 = (den * (Y * X2 - Y2 * float(n)) + (Y1 * float(n) - Y * X) * (X4 * float(n) - X2 * X2))

        b = temp2 / temp
        a = 1.0 / den * (float(n) * Y1 - Y * X - b * (X2 * float(n) - X * X))
        c = 1 / float(n) * (Y - a * X2 - b * X)

        k = 0;
        SSE = 0.0;
        SST = 0.0;

        f = np.zeros(len(x), float)

        while k < len(x):
            f[k] = a * x[k] ** 2 + b * x[k] + c
            SSE = SSE + (y[k] - f[k]) ** 2
            SST = SST + (y[k] - m) ** 2
            k = k + 1
        R2 = 1 - SSE / SST

        return [a, b, c, R2, SSE, SST]

    def _FHFP(self, w1, gas2, gas1):
        """
        Uses the momentum and energy conservation equations to calculate
        error in pressure and enthalpy given shock speed, upstream (gas1)
        and downstream states (gas2).  States are not modified by these routines.

        FUNCTION SYNTAX:
            [FH,FP] = FHFP(w1,gas2,gas1)

        INPUT:
            w1 = shock speed (m/s)
            gas2 = gas object at working/downstream state
            gas1 = gas object at initial/upstream state

        OUTPUT:
            FH,FP = error in enthalpy and pressure

        """
        P1 = gas1.P
        H1 = gas1.enthalpy_mass
        r1 = gas1.density
        P2 = gas2.P
        H2 = gas2.enthalpy_mass
        r2 = gas2.density
        w1s = w1 ** 2
        w2s = w1s * (r1 / r2) ** 2
        FH = H2 + 0.5 * w2s - (H1 + 0.5 * w1s)
        FP = P2 + r2 * w2s - (P1 + r1 * w1s)
        return [FH, FP]

    def _eq_state(self, gas, r1, T1):
        """
        Calculates equilibrium state given T & rho.
        Used in postshock module.

        FUNCTION SYNTAX:
            [P,H] = eq_state(gas,r1,T1)

        INPUT:
            gas = working gas object (gas object is changed and corresponds to new equilibrium state)
            r1,T1 = desired density and temperature

        OUTPUT
            P,H = equilibrium pressure and enthalpy at given temperature and density

        """
        gas.TD = T1, r1
        gas.equilibrate('TV')
        P = gas.P
        H = gas.enthalpy_mass
        return [P, H]

def create_chapman_jouguet_analyzer(mech: Any, solver_type: str = "manual", verbose: bool = False) -> ChapmanJouguetAnalyzer:
    """Factory function to create a Chapman-Jouguet analyzer instance.

    Args:
        mech (Any): Mechanism file or object.
        solver_type (str): Type of solver to use ('manual', 'SDToolbox', 'compare').
        verbose (bool): If True, enables verbose output.

    Returns:
        ChapmanJouguetAnalyzer: An instance of a CJ analyzer.
    """
    return CJAnalyzer(mech, solver_type=solver_type, verbose=verbose)

if __name__ == "__main__":
    # Set initial thermodynamic state
    T = 300.0  # Initial temperature in K
    P = 101325.0  # Initial pressure in Pa
    mech = '../../../chemical_mechanisms/LiDryer.yaml'  # Path to mechanism file
    phi_array = np.linspace(0.2, 2.0, 100)

    # Create analyzer using factory function
    analyzer = create_chapman_jouguet_analyzer(mech, verbose=False)

    def_cj_results = []
    det_cj_results = []

    for phi in phi_array:
        # Stoichiometric hydrogen-oxygen mixture
        q = {'H2': 2 * phi, 'O2': 1, 'N2': 3.76}  # Adjust for equivalence ratio

        print(f"\nCalculating for equivalence ratio φ={phi:.2f}")

        # Deflagration calculation
        if phi > 1:
            print("WARNING: Rich mixtures may not support CJ deflagration (W.I.P.)")
        try:
            def_result = analyzer.analyze_cj_deflagration(T, P, q)
            def_cj_results.append(def_result)
            print(f'Deflagration: v={def_result.velocity:.1f} m/s, T={def_result.temperature:.1f} K')
        except Exception as e:
            print(f'Deflagration failed: {e}')
            # Add placeholder result
            def_cj_results.append(None)

        # Detonation calculation
        try:
            det_result = analyzer.analyze_cj_detonation(T, P, q)
            det_cj_results.append(det_result)
            print(f'Detonation: v={det_result.velocity:.1f} m/s, T={det_result.temperature:.1f} K')
        except Exception as e:
            print(f'Detonation failed: {e}')
            det_cj_results.append(None)

    # Extract velocities for plotting
    def_cj_vel = [r.velocity if r else np.nan for r in def_cj_results]
    det_cj_vel = [r.velocity if r else np.nan for r in det_cj_results]

    # Plot results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('CJ solutions vs equivalence ratio φ', fontsize=14)

    # Note: axes is 1-D here (2 rows, 1 col)
    ax0, ax1 = axes

    # (1) Detonation CJ speed
    ax0.plot(phi_array, det_cj_vel, 'k-', label='CJ Detonation Speed', linewidth=2)
    ax0.set_ylabel('CJ Detonation Velocity [m/s]')
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    # (2) Deflagration CJ speed
    ax1.plot(phi_array, def_cj_vel, 'b-', label='CJ Deflagration Speed', linewidth=2)
    ax1.set_xlabel('Equivalence ratio φ [-]')
    ax1.set_ylabel('CJ Deflagration Velocity [m/s]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()