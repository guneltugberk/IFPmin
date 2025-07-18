from IFPmin import CO2InjectionConfig
import CoolProp.CoolProp as CP
from scipy.integrate import quad
import math
import numpy as np

class GasProperties:
    """
    Extends CO2InjectionConfig with CO2 PVT methods (CoolProp).
    """
    def __init__(self, cfg: CO2InjectionConfig):

        self.cfg = cfg
        rp   = self.cfg.config['resprop']
        w    = self.cfg.config['wellspec']

        self.temperature = rp['temperature_C']
        self.pressure    = rp['pressure_bar']

        self.porosity_min, self.porosity_avg, self.porosity_max = (
            rp['petro']['porosity_min'],
            rp['petro']['porosity_avg'],
            rp['petro']['porosity_max'],
        )
        self.c_r  = rp['rock_compressibility_1_per_bar']
        self.c_w  = rp['water_compressibility_1_per_bar']
        self.Pinj = w['injection_pressure_bar']
        self.Pfrac= rp['frac_pressure_bar']
        self.rw   = w['wellbore_radius_m']

        # convert to SI
        self.T_K  = self.temperature + 273.15
        self.P_pa = self.pressure * 1e5

    def co2_density(self):
        """
        Return the density of CO₂ at the configured P & T.
        Units: kg/m3.
        """
        return CP.PropsSI('D', 'P', self.P_pa, 'T', self.T_K, 'CO2')

    def co2_viscosity(self) -> float:
        """
        Return the viscosity of CO₂ at the configured P & T.
        Units: cP.
        """
        mu = CP.PropsSI('V', 'P', self.P_pa, 'T', self.T_K, 'CO2')

        return mu * 1e3  # Pa·s → cP

    def co2_z_factor(self) -> float:
        """
        Return the gas deviation factor of CO₂ at the configured P & T.
        Units: Dimensionless.
        """
        return CP.PropsSI('Z', 'P', self.P_pa, 'T', self.T_K, 'CO2')

    def co2_volume_factor(self) -> float:
        """
        Return the formation volume factor of CO₂ at the configured P & T.
        Units: Dimensionless.
        """
        return (101325 / 298.15) * (self.co2_z_factor() * self.T_K) / self.P_pa
    
    def co2_compressibility(self) -> float:
        """
        Return the isothermal compressibility of CO₂ at the configured P & T.
        Units: 1/bar.
        """
        # get in 1/Pa
        kappa_pa = CP.PropsSI(
            'ISOTHERMAL_COMPRESSIBILITY',
            'P', self.P_pa,
            'T', self.T_K,
            'CO2'
        )

        # convert to 1/bar
        kappa_bar = kappa_pa * 1e5

        return kappa_bar

    def real_gas_pseudo_pressure(self, Pinj=None):
        Pinj = (Pinj or self.Pinj) * 14.5038  # bar→psi
        Pres = self.pressure * 14.5038

        def integrand(p):
            mu = self.co2_viscosity()
            z  = self.co2_z_factor()
            return 2 * p / (mu * z)
        val, _ = quad(integrand, Pinj, Pres, limit=1000)

        return round(val, 2)
    
class IFlowP(GasProperties):
    """
    Extension of GasProperties to handle injection-pressure related calculations.
    """
    def __init__(self, cfg: CO2InjectionConfig):
        super().__init__(cfg)


    def co2_injectivity(self, poro, k, h, rw):
        """
        Compute single-phase gas injection rate or injectivity index.

        Args:
            k (float): Permeability in mD.
            h (float): Reservoir thickness in meters.
            rw (float): Wellbore radius in meters.
            skin (float): Dimensionless skin factor.
            display_results (bool): Displaying the simulation output. Default is True.

        Returns:
            J: Approximate injectivity index (tonnes CO₂/${bar^2}$.cp.year).
        """

        mu_co2_cp = self.co2_viscosity()                                                # [cp]
        rho_co2   = self.co2_density()                                                  # [kg/m³]
        T         = self.temperature                                                    # [°C]
        p_init    = self.pressure                                                       # [bar]

        # Convert k md → m²
        k_SI = k * 9.869233e-16                                                         # [m²]

        # CO2 gas compressibility 
        c_g = self.co2_compressibility()                                                # [1/bar]

        # Total compressibility
        c_t = self.c_r + self.c_w * 0.5 + c_g * (1 - 0.5)                               # [1/bar]

        # Simulation time conversions
        DAYS_PER_MONTH = 30.4375
        SECONDS_PER_DAY = 86400
        SECONDS_PER_MONTH = DAYS_PER_MONTH * SECONDS_PER_DAY

        # Simulation time frame
        delta_t_months = (self.cfg.config['target']['time_start_yr'] - self.cfg.config['target']['time_end_yr'])                                                    # [months]

        # ------------------------------------------
        # 2. Non-Darcy Affect Injectivity Approach
        # ------------------------------------------
        mu_g   = mu_co2_cp                                                              # [cp]
        gamma  = 1.7842 / 1.293  # Specific gravity of CO2                              # [-]
        delta_psi = abs(self.real_gas_pseudo_pressure(self.Pinj))                       # [psi^2/cp]

        # Convert to field units
        T_rankine = (9.0 / 5.0) * T + 491.67                                            # [R]
        h_ft = 3.28084 * h                                                              # [ft]
        rw_ft = 3.28084 * self.rw                                                       # [ft]
        kh = k * h_ft                                                                   # [md.ft]

        # Beta correlation from (7.116b) of Petrophysics Handbook
        beta = (4.85e4) / ((poro ** 5.5) * (k ** 0.5))                     # [1/ft]

        # Non-Darcy/turbulent coefficient from (7.138) of Petrophysics Handbook
        D = ((2.22e-15 * gamma) / (mu_g * rw_ft * h_ft)) * beta * k                     # [-]

        # Quadratic eqn: Q^2 * b2 + Q * a2 - delta_psi = 0
        a2 = ((1422.0 * T_rankine) / kh) * (7 + 0)                                      # 0 indicates skin -- will be changed in the future
        b2 = ((1422.0 * T_rankine) / kh) * D

        inside_sqrt = a2**2 + (4.0 * b2 * delta_psi)

        if inside_sqrt < 0:
            raise ValueError("Negative sqrt => no real solution for Qg.")
        
        Qg_turb = (-a2 + math.sqrt(inside_sqrt)) / (2.0 * b2)                           # [MSCF/D]

        if Qg_turb < 0:
            raise ValueError(f"Solved Qg <= 0 => non-physical. Qg={Qg_turb}")

        # Effective skin
        effective_skin = 0 + Qg_turb * D                                               # [-]

        # Convert delta_psi (psi^2/cp) → bar^2/cp
        delta_p_bar2 = delta_psi * 0.0689476 * 0.0689476                                # [bar^2/cp]

        # Convert Qg from MSCF/D → tonnes/year of CO2
        Qg = (28.316846592 * Qg_turb * 1.7842 / 1000) * 365                             # [tonnes CO2/year]

        # Injectivity index
        II = Qg / delta_p_bar2                                                          # [tonnes CO2.cp / (bar².year)]

        # -------------------------------------
        # 3. Computing Total Injection Volume
        # -------------------------------------

        # Convert dynamic viscosity cP → Pa·s
        mu_co2_SI = mu_co2_cp * 1.0e-3                                                  # [Pa.s]
        c_t_psi = c_t * 0.0689475729                                                    # 1/psi

        # Constants
        GAMMA = 1.781                                                                   # Euler's constant

        def dimensionless_time(t):
            # Dimensionless time from Equation 4.19 (Practice of RE, L.P. Dake)
            tD = 0.000264 * (k * t) / (poro * mu_co2_cp * c_t_psi * (rw_ft ** 2))

            return tD

        def dimensionless_pressure(t):
            # p_D = 0.5 ln(4 t_D / gamma) from Equation 6
            t_d = dimensionless_time(t)
            pD = 0.5 * np.log((4 * t_d) / GAMMA) + 0
            
            return pD
        
        def MDH(t, dt):
            t_hours = t * DAYS_PER_MONTH * 24 # hours
            delta_t_hours = dt * DAYS_PER_MONTH * 24 # hours
        
            pD = dimensionless_pressure(t_hours)

            term = (4 * 0.000264 * k) / (GAMMA * poro * mu_co2_cp * c_t_psi * rw_ft ** 2)
            C = pD - 1.151 * np.log10(term)
            eq34 = C - 1.151 * np.log10(delta_t_hours)

            return eq34

        
        t_f_s = self.cfg.config['target']['time_end_yr'] * SECONDS_PER_MONTH
        t_i_s = self.cfg.config['target']['time_start_yr'] * SECONDS_PER_MONTH

        time_span = np.linspace(start=self.cfg.config['target']['time_start_yr'], stop=self.cfg.config['target']['time_end_yr'], num=500)
        delta = np.array([time_span[i+1] - time_span[i] for i in range(len(time_span)-1)])

        delta_t_hours = delta * DAYS_PER_MONTH * 24 # hours

        # Constant term in the integral
        constant = 4 * k_SI / (GAMMA * poro * mu_co2_SI * c_t * (rw ** 2))

        # Delta time of the project                                                     # months
        delta_t_months = (self.cfg.config['target']['time_end_yr'] - self.cfg.config['target']['time_start_yr']) * 12

        # First term of the integral, pseudo pressure difference
        term1_int = (delta_p_bar2) * DAYS_PER_MONTH * delta_t_months                    # Convert months to days | bar^2·day/cp

        if self.cfg.config['target'].get('time_start_yr', 0) < 0:
            raise ValueError("Initial time t_i must be greater than 0 to avoid logarithm of zero.")
        
        term_integral = (4 * 0.000264 * k) / (GAMMA * poro * mu_co2_cp * c_t_psi * rw_ft ** 2)
        
        integral_ln_tf =  (t_f_s / 3600) * (math.log(constant * t_f_s / 3600) - 1)      # hours
        C_f = integral_ln_tf - 1.151 * np.log10(term_integral)                          # hours
        integral_ln_tf_pbu = C_f - 1.151 * np.log10(delta_t_hours[-1])                  # hours

        if self.cfg.config['target'].get('time_start_yr', 0) == 0:
            integral_ln_ti_pbu = 0
        
        else:
            integral_ln_ti = (t_i_s / 3600) * (math.log(constant * t_i_s / 3600) - 1)   # hours
            C_i = integral_ln_ti - 1.151 * np.log10(term_integral)                      # hours
            integral_ln_ti_pbu = C_i - 1.151 * np.log10(delta_t_hours[-1])              # hours

        integral_ln = (integral_ln_tf_pbu - integral_ln_ti_pbu) * 3600                  # seconds
        term2_init = (0.5 * (self.cfg.config['injsim_params'].get('A') ** 2) * integral_ln) / (86400 * mu_co2_cp)               # Convert seconds to days | bar^2·day/cp

        # Total integral value (bar·days)
        integral_total = term1_int + term2_init                                         # bar^2·day/cp

        # [tonne CO2·cp / (bar²·year)] → m³·cp / (bar²·day)
        I_c_m3_per_bar2_cp_day = ((II / 365.0) * 1000.0 / 1.7842) 

        # ------------------------------------------
        # 4. Total injection volume (Equation 8)
        # ------------------------------------------
        V = (I_c_m3_per_bar2_cp_day * integral_total + self.cfg.config['injsim_params']['Fb']) * (rho_co2 / 1000) 

        # ------------------------------------------
        # 5. Final reservoir pressure (Equation 7)
        # ------------------------------------------
        time_span = np.delete(time_span, 0)
        p_res = p_init + self.cfg.config['injsim_params'].get('A') * MDH(t=time_span, dt=delta)

        # -----------------------------
        # 6. Well count from V
        # -----------------------------
        self.wells = math.ceil(self.cfg.config['target'].get('storage_objective_t_per_year') / Qg)                                              # [-]
        self.sufficient = self.cfg.config['target'].get('storage_objective_t_per_year') / V 

        return II, V, self.wells, delta_p_bar2, Qg, effective_skin, self.sufficient, p_res, time_span

