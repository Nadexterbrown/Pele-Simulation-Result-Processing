from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cantera as ct

from ..core.domain import CJProperties, CJType
from ..core.interfaces import ChapmanJouguetAnalyzer


class CJAnalyzer(ChapmanJouguetAnalyzer):

    def __init__(self, mech: Any, solver_type: str = "manual", verbose: bool = False):
        self.mech = mech
        self.solver_type = solver_type
        self.verbose = verbose

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
            return self._cj_deflagration_solver(T, P, Y)
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
        self.wave = CJType.DETONATION
        return self._cj_detonation_solver(T, P, Y)

    def _cj_deflagration_solver(self, T: float, P: float, q: Any) -> CJProperties:
        from sdtoolbox.postshock import eq_state, FHFP, LSQ_CJspeed

        # Try to import soundspeed functions, use fallback if not available
        try:
            from sdtoolbox.thermo import soundspeed_eq, soundspeed_fr
        except (ImportError, AttributeError):
            # Fallback implementations using Cantera
            def soundspeed_eq(gas):
                """Compute equilibrium sound speed using Cantera."""
                return gas.sound_speed

            def soundspeed_fr(gas):
                """Compute frozen sound speed using Cantera."""
                gamma = gas.cp / gas.cv
                return np.sqrt(gamma * ct.gas_constant * gas.T / gas.mean_molecular_weight)

        def _perfect_gas_properties(P0, T0, mix, mech):
            """

        	perfect_gas_properties
            Uses Cantera to estimate perfect gas properties (gamma, rgas, Qcomb)
            at initial state defined by P0, T0, mix.

            SYNTAX
            [gamma, rgas, Qgas] = perfect_gas_properties(P0, T0, mix, mech)

            INPUT
            P0 = Initial pressure (Pa)
            T0 = Initial temperature (K)
            mix = string of reactant species mole fractions
            mech= sandiego2014.cti or gri30.cti for example

            OUTPUT
            gamma = Heat capacity ratio
            rgas = specific perfect gas constant
            Qgas = energy of reaction for the gas

            """

            # Set gas at standard state
            gas = ct.Solution(mech)
            gas.TPX = T0, P0, mix

            # Extract fresh gases properties
            rgas = ct.gas_constant / gas.mean_molecular_weight
            gamma = gas.cp_mass / gas.cv_mass
            uf0 = gas.int_energy_mass

            # Equilibrate at UV constant and bring to T0, P0
            gas.equilibrate("UV")
            gas.TP = T0, P0
            ub0 = gas.int_energy_mass

            # Calculate heat of reaction per kg of fuel
            Qgas = (uf0 - ub0)

            return [gamma, rgas, Qgas]

        def _cj_deflagration_perfect(P1, T1, gamma1, rgas, Qgas):

            def _mcj_def(Pi, vi, Q, gamma):
                """

            	Mcj_def
                Calculates the CJ deflagration Mach number in the perfect gas approximation

                SYNTAX
                Mcj = Mcj_def(Pi, vi, Q, gamma)

                INPUT
                Pi : Initial pressure [Pa]
                vi : Initial volume [m3]
                Q : Energy of combustion of the mixture [J/kg of mixture], estimated for a perfect gas
                gamma : specific heat ratio of the gas [-], estimated for a perfect gas.

                OUTPUT
                Mcj : Deflagration Mach number estimated for a perfect gas

                NOTES
                - gamma can be estimated using function "perfect_gas_properties" that returns
                  the perfect gas properties of a mixture at a given state.

                """
                n = (gamma ** 2.0 - 1) / gamma * (Q / (Pi * vi))
                return (1 + n - ((n + 1) ** 2.0 - 1) ** 0.5) ** 0.5

            """

            CJ_deflagration_perfect
            Calculates the CJ_deflagration speed in the approximation of a perfect gas.
            Uses Cantera to estimate perfect gas properties (gamma, Qcomb, rgas).

            SYNTAX
            [Dcj, Pcj, Tcj, vcj, Mflow_cj] = CJ_deflagration_perfect(P1, T1, gamma1, rgas, Qgas)

            INPUT
            P1 = Initial pressure (Pa)
            T1 = Initial temperature (K)
            gamma1 = Specific heat ratio
            rgas = Perfect gas constant of the gas
            Qgas = Energy of reaction ofhte gas

            OUTPUT
            Dcj = CJ deflagration velocity
            Pcj = post CJ deflagration pressure
            Tcj = post CJ deflagration temperature
            vcj = post CJ deflagration specific volume
            Mflow_cj =  Mach Number of the flow at the post CJ deflagration state

            """

            # Define initial state
            rho1 = P1 / (rgas * T1)
            v1 = 1 / rho1
            c1 = (gamma1 * rgas * T1) ** 0.5

            # Calculate CJ deflagration speed
            Mcj = _mcj_def(P1, v1, Qgas, gamma1)

            # Calculate vcj and Pcj
            vcj = v1 * (1 + gamma1 * Mcj ** 2) / (Mcj ** 2 * (1 + gamma1))
            Pcj = P1 * (1 + gamma1 * Mcj ** 2) / (1 + gamma1)

            # Determine the rest of variables
            Tcj = Pcj * vcj / rgas
            ccj = (gamma1 * rgas * Tcj) ** 0.5

            Dcj = Mcj * c1
            uflow_cj = Dcj * vcj / v1
            Mflow_cj = uflow_cj / ccj

            return [Dcj, Pcj, Tcj, vcj, Mflow_cj]

        def _cj_calc(gas, gas1, ERRFT, ERRFV, x, w1_guess=500, T_guess=1000):

            """
            CJ_calc3
            For a reactive discontinuity, this function calculate for a given specific volume
            (represented by the ratio x = v1/v2), the velocity of the discontinuity and the
            state downstream it.
            This is equivalent to find the point on the Hugoniot curve and the velocity assocuated
            at a given v2 from a known initial state.

            FUNCTION
            SYNTAX
            [gas,w1,T] = CJ_calc3(gas,gas1,ERRFT,ERRFV,x)

            INPUT
            gas = working gas object
            gas1 = gas object at initial state
            ERRFT,ERRFV = error tolerances for iteration
            x = volume ratio(v1/v2)

            OUTPUT
            gas = gas object at equilibrium state
            w1 = initial velocity to yield prescribed density ratio
            T =  temperature of the downstream state
            """

            r1 = gas1.density
            V1 = 1 / r1;
            P1 = gas1.P;
            T1 = gas1.T
            i = 0;
            DT = 1000;
            DV = 1000;
            DP = 1000;
            DW = 1000

            # PRELIMINARY GUESS
            V = V1 / x;
            r = 1 / V;
            w1 = w1_guess;
            T = T_guess;
            [P, H] = eq_state(gas, r, T)

            # START LOOP
            while (abs(DT) > ERRFT * T and abs(DW) > ERRFV * w1):
                i = i + 1
                if i == 500:
                    print
                    'CJ_calc function could not converge after 500 iterations on the Newton-Raphson algorithm.'
                    print
                    "Results given with : abs(DT) - ERRFT*T = %e \t abs(DW) - ERRFW*w1 = %e" \
                    % (abs(DT) - ERRFT * T, abs(DW) - ERRFV * w1)
                    break
                    # return

                # CALCULATE FH & FP FOR GUESS 1
                [FH, FP] = FHFP(w1, gas, gas1)

                # TEMPERATURE PERTURBATION
                DT = T * 0.02;
                Tper = T + DT;
                Vper = V;
                Rper = 1 / Vper;
                Wper = w1;
                [Pper, Hper] = eq_state(gas, Rper, Tper)
                # CALCULATE FHX & FPX FOR "IO" STATE
                [FHX, FPX] = FHFP(Wper, gas, gas1)
                # ELEMENTS OF JACOBIAN
                DFHDT = (FHX - FH) / DT;
                DFPDT = (FPX - FP) / DT;

                # VELOCITY PERTURBATION
                DW = 0.02 * w1;
                Wper = w1 + DW;
                Tper = T;
                Rper = 1 / V;
                [Pper, Hper] = eq_state(gas, Rper, Tper)
                # CALCULATE FHX & FPX FOR "IO" STATE
                [FHX, FPX] = FHFP(Wper, gas, gas1)
                # ELEMENTS OF JACOBIAN
                DFHDW = (FHX - FH) / DW;
                DFPDW = (FPX - FP) / DW;

                # INVERT MATRIX
                J = DFHDT * DFPDW - DFPDT * DFHDW
                b = [DFPDW, -DFHDW, -DFPDT, DFHDT]
                a = [-FH, -FP]
                DT = (b[0] * a[0] + b[1] * a[1]) / J;
                DW = (b[2] * a[0] + b[3] * a[1]) / J;

                # CHECK & LIMIT CHANGE VALUES
                # VOLUME
                DTM = 0.2 * T
                if abs(DT) > DTM:
                    DT = DTM * DT / abs(DT)
                # MAKE THE CHANGES
                T = T + DT;
                w1 = w1 + DW;
                [P, H] = eq_state(gas, r, T)

            return gas, w1, T

        def _cj_speed(P1, T1, mix, mech, xmin, xmax, plt_num, ERRFT = 1e-6, ERRFV = 1e-6, w1_guess=500, T_guess=1000):
            """

                CJspeed3
                Calculates CJ deflagration velocity.
                WR: This method is based on the minimum wave speed algorithm

                FUNCTION
                SYNTAX
                [cj_speed,R2,dnew] = CJspeed3(P1,T1,mix,mech,xmin,xmax,plt_num)

                INPUT
                P1 = initial pressure (Pa)
                T1 = initial temperature (K)
                mix = string of reactant species mole fractions
                mech = cti file containing mechanism data (i.e. 'gri30.cti')
                plt_num = unused
                xmin = minimum of volume ratio
                xmax = maximum of volume ratio
                OUTPUT
                cj_speed = CJ deflagration speed (m/s)
                R2 = R-squared value of LSQ curve fit
                dnew = volume ratio at CJ state
            """
            # DECLARATIONS
            numsteps = 20;
            maxv = xmax;
            minv = xmin;

            w2 = np.zeros(numsteps, float)
            rr = np.zeros(numsteps, float)

            gas1 = ct.Solution(mech)
            gas = ct.Solution(mech)

            # INTIAL CONDITIONS
            gas.TPX = T1, P1, mix
            gas1.TPX = T1, P1, mix

            # INITIALIZE ERROR VALUES & CHANGE VALUES
            # ERRFT = 1*10**-6;  ERRFV = 1*10**-6;

            i = 1;
            T1 = gas1.T;
            P1 = gas1.P;

            R2 = 0.0;
            cj_speed = 0.0
            a = 0.0;
            b = 0.0;
            c = 0.0;
            dnew = 0.0

            # Set parameters for the loop
            counter = 1;  # Initialize counter
            mincounter = 4;  # Minimum loop iteration
            maxcounter = 20;  # Maximum loop iteration

            # Start loop to find the maximum flame speed
            while (counter <= mincounter) or (R2 < 0.99999):
                step = (maxv - minv) / float(numsteps)
                i = 0
                x = minv

                # Calculate several valules of w1 and v1/v2 on the interval [minv;maxv]
                for i in range(0, numsteps):
                    gas.TPX = T1, P1, mix

                    # Call subfunction to find speed for a specific ratio of volume
                    [gas, temp, T] = _cj_calc(gas, gas1, ERRFT, ERRFV, x, w1_guess, T_guess=T_guess)
                    w2[i] = temp
                    rr[i] = gas.density / gas1.density

                    i = i + 1;
                    x = x + step;

                # Fit the points of graph w1 = f(v1/v2) to a parabola.
                [a, b, c, R2, SSE, SST] = LSQ_CJspeed(rr, w2)

                # Get the locus of maximum(v1/v2) = dnew to be the CJ specific volume.
                dnew = -b / (2.0 * a)

                # Define the a new interval where to find maximum flame speed as dnew (+-) 1%.
                minv = dnew - dnew * 0.001
                maxv = dnew + dnew * 0.001

                # Get the current estimation of CJ-deflagration speed being the maximum of value
                # of the parabolic fit.
                cj_speed = a * dnew ** 2 + b * dnew + c

                # Update the counter and re-run the loop until R2 for parabolla fit small enough.
                counter = counter + 1;

                # Stop the loop on maximum speed if too much iterations, and warn user.
                if counter > maxcounter:
                    print
                    "\nReached %i iteration on finding maximum flame speed in CJSpeed3 function." % (counter)
                    print
                    "Residual on least-square algorithm R2 = ", R2, "\n"
                    break

            return cj_speed, R2, dnew

        """
        CJ_deflagration
        Calculates CJ speed for a deflagration , gas at CJ state and Mach Number.

        FUNCTION
        SYNTAX
        [ucj,gas,M] = CJ_deflagration(P1,T1,mix,mech)

        INPUT
        P1 = Initial pressure (Pa)
        T1 = Initial temperature (K)
        mix = string of reactant species mole fractions
        mech = sandiego2014.cti or gri30.cti for example

        OPTIONAL ARGUMENTS
        ERRFT = 1e-6 : Tolerance on temperature for the 2-variables (T,v) Newton-Raphson algorithm [K]
        ERRFT = 1e-6 : Tolerance on volume for the 2-variables (T,v) Newton-Raphson algorithm [m3/kg]

        OUTPUT
        ucj = CJ deflagration
        gas = gas object at CJ state
        M =  Mach Number
        """

        ERRFT = 1e-6
        ERRFV = 1e-6

        # Initialize initial gas at  state T1, P1, mix (input)
        gas1 = ct.Solution(self.mech)
        gas1.TPX = T, P, q

        # Estimate CJ-deflagration volume for perfect gas, and volume ratio associated.
        # This is to be used to find the minimum volume ratio of the interval where to find
        # the CJ-deflagration volume.
        [gamma1, rgas, Qgas] = _perfect_gas_properties(ct.one_atm, 298.15, q, self.mech)
        [Dcj_id, Pcj_id, Tcj_id, vcj_id, mflowcj_id] = _cj_deflagration_perfect(P, T, gamma1, rgas, Qgas)

        rho1_id = P / (rgas * T)
        v1_id = 1 / rho1_id
        xcj_id = v1_id / vcj_id

        # Define the minimum volume ratio of the interval where to find the CJ-deflagration
        # volume, as (xcj_id - 50% of xcj_id), equivalent to take vmax = 2*vcj_id
        xmin = xcj_id * (1 - 0.5)

        # Calculate the constant pressure combustion, that will give the volume of the
        # post-constant pressure combustion state.
        gas_CP = ct.Solution(self.mech)
        gas_CP.TPX = T, P, q
        gas_CP.equilibrate('HP')

        # Define the maximum volume ratio of the interval where to find the CJ-deflagration
        # volume, as the one given for CP-combustion volume (minus 1% for safety).
        xmax = (gas1.v / gas_CP.v) * (1 - 0.01)

        # Call function CJspeed3 to find CJ deflagration velocity and x_cj the
        # volume ratio at the CJ deflagration state
        # Uses w1_guess = Dcj_id + 50% to favorise convergence from the top
        [D_cj, R2, x_cj] = _cj_speed(P, T, q, self.mech, xmin, xmax, 0, ERRFT, ERRFV, w1_guess=Dcj_id * 1.5,
                                    T_guess=Tcj_id * 0.8)

        # Calculate gas_cj a gas object at CJ deflagration state
        gas_cj = ct.Solution(self.mech)
        gas_cj.TPX = T, P, q
        [gas_cj, temp, T_cj] = _cj_calc(gas_cj, gas1, ERRFT, ERRFV, x_cj, w1_guess=Dcj_id * 1.5, T_guess=Tcj_id * 0.8)

        # Verify gas_cj is still a gas object
        if not isinstance(gas_cj, ct.Solution):
            raise TypeError(f"Expected gas_cj to be a Cantera Solution object, but got {type(gas_cj)}")

        # Calculate the equilibrium sound speed in the CJ-deflagration state to get the
        # Mach number of the flow in the burnt gases
        try:
            c_cj_eq = soundspeed_eq(gas_cj)
            c_cj_fr = soundspeed_fr(gas_cj)
        except Exception as e:
            # Fallback to Cantera's built-in sound speed
            c_cj_eq = gas_cj.sound_speed
            gamma = gas_cj.cp / gas_cj.cv
            c_cj_fr = np.sqrt(gamma * ct.gas_constant * gas_cj.T / gas_cj.mean_molecular_weight)

        # Calculate flow speed and Mach number in the CJ-deflagration state
        uflow_cj = D_cj / x_cj
        Mflow_cj = uflow_cj / c_cj_eq


        return CJProperties(
            cj_type=self.wave,
            pressure=gas_cj.P,
            temperature=gas_cj.T,
            density=gas_cj.density,
            reactant_velocity=D_cj,
            product_velocity=uflow_cj,
            specific_energy=gas_cj.int_energy_mass,
            specific_volume=1.0 / gas_cj.density_mass,
            sound_speed=gas_cj.sound_speed,
            gamma=gas_cj.cp_mass / gas_cj.cv_mass,
            mach_number=uflow_cj / gas1.sound_speed,
            product_mach_number=Mflow_cj,
            density_ratio=x_cj,
            enthalpy=gas_cj.enthalpy_mass,
            r_squared=R2,
            converged=R2 > 0.999,
            species_concentrations=gas_cj.X
        )

    def _cj_detonation_solver(self, T: float, P: float, q: Any) -> CJProperties:
        try:
            from sdtoolbox.postshock import CJspeed, PostShock_eq
        except ImportError:
            raise ImportError("SDToolbox not installed. Please install it to use CJ detonation analysis.")

        # Initialize initial gas at  state T1, P1, mix (input)
        gas_initial = ct.Solution(self.mech)
        gas_initial.TPX = T, P, q

        # Calculate CJ Detonation using SDToolbox
        [cj_speed, R2, plot_data] = CJspeed(P, T, q, self.mech, fullOutput=True)

        # compute equilibrium CJ state parameters
        gas_final = PostShock_eq(cj_speed, P, T, q, self.mech)

        return CJProperties(
            cj_type=self.wave,
            pressure=gas_final.P,
            temperature=gas_final.T,
            density=gas_final.density,
            reactant_velocity=cj_speed,
            product_velocity=cj_speed * gas_initial.density_mass / gas_final.density_mass,
            specific_energy=gas_final.int_energy_mass,
            specific_volume=1.0 / gas_final.density_mass,
            sound_speed=gas_final.sound_speed,
            gamma=gas_final.cp_mass / gas_final.cv_mass,
            mach_number=cj_speed / gas_initial.sound_speed,
            product_mach_number=(cj_speed*gas_initial.density_mass/gas_final.density_mass),
            density_ratio=gas_initial.density_mass/gas_final.density_mass,
            enthalpy=gas_initial.enthalpy_mass,
            r_squared=R2,
            converged=R2 > 0.999,
            species_concentrations=gas_final.X
        )


def create_chapman_jouguet_analyzer(mech: Any, solver_type: str = "manual", verbose: bool = False) -> CJAnalyzer:
    """Factory function to create a Chapman-Jouguet analyzer instance.

    Args:
        mech: Cantera mechanism file path or object
        solver_type: Type of solver to use ("manual" or other implementations)
        verbose: Enable verbose output during calculations

    Returns:
        CJAnalyzer: Configured Chapman-Jouguet analyzer instance
    """
    return CJAnalyzer(mech=mech, solver_type=solver_type, verbose=verbose)