import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.integrate import quad

class IFPMin:
    """
    Base class for CO2 storage capacity calculation and simulation data handling.
    """

    # --- Constants ---
    CO2_MOLECULAR_WEIGHT = 44.01       # g/mol
    DEFAULT_TEMPERATURE = 40.0         # °C
    DEFAULT_PRESSURE = 249             # bar
    DEFAULT_POROSITY = 0.089           # fraction
    DEFAULT_POROSITY_MAX = 0.2         # fraction
    DEFAULT_SW_MIN = 0.30              # fraction
    DEFAULT_SW_MAX = 0.95              # fraction

    def __init__(
        self,
        input_data=None,
        simulation_result=None,
        target=30e9,
        include_dolo=True,
        temperature=None,
        pressure=None,
        porosity=None,
        porosity_max=None,
        sw_min=None,
        sw_max=None,
        seed=141
    ):
        """
        Initialize the CO2 storage capacity calculator.

        Args:
            input_data (pd.DataFrame, optional): Input mineral dataset.
            simulation_result (pd.DataFrame, optional): Precipitated minerals dataset.
            target (float): Storage target of CO2 in kg. Default 3e10.
            include_dolo (bool): Whether to include dolomite.
            temperature (float, optional): Temperature (°C).
            pressure (float, optional): Pressure (bar).
            porosity (float, optional): Average porosity fraction.
            porosity_max (float, optional): Maximum porosity fraction.
            sw_min (float, optional): Minimum water saturation fraction.
            sw_max (float, optional): Maximum water saturation fraction.
            seed (int): Random seed for reproducibility.
        """

        self.input_data = input_data
        self.simulation_result = simulation_result
        self.target = target
        self.include_dolo = include_dolo
        self.seed = seed
        np.random.seed(self.seed)

        # Set defaults if not provided
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.pressure = pressure if pressure is not None else self.DEFAULT_PRESSURE
        self.porosity = porosity if porosity is not None else self.DEFAULT_POROSITY
        self.porosity_max = porosity_max if porosity_max is not None else self.DEFAULT_POROSITY_MAX
        self.sw_min = sw_min if sw_min is not None else self.DEFAULT_SW_MIN
        self.sw_max = sw_max if sw_max is not None else self.DEFAULT_SW_MAX

        self.calculate_co2 = False  # Will be set True after process_simulation_data

    def update_simulation_conditions(self, temperature=None, pressure=None):
        """
        Update the simulation conditions dynamically.

        Args:
            temperature (float, optional): New temperature for the simulation (°C).
            pressure (float, optional): New pressure for the simulation (bar).
        """
        
        if temperature is not None:
            self.temperature = temperature
            print(f"Updated temperature to {self.temperature} °C.")
            
        if pressure is not None:
            self.pressure = pressure
            print(f"Updated pressure to {self.pressure} bar.")

    def generate_simulation_files(self, output_dir="Simulation", num_simulation=1000):
        """
        Generate simulation input files (.inn) from a dataset for ArXim.

        Args:
            output_dir (str): Directory to store the simulation input files.
            num_simulation (int): Number of random simulations to generate.

        Returns:
            str: The output directory where the files are saved.
        """

        if self.input_data is None:
            raise ImportError("Input data cannot be None.")
        if self.simulation_result is None:
            raise ImportError("Simulation result data cannot be None.")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Template for the .inn input file (ArXim simulation)
        input_template = """
TEST
COMPUTE SPC
COMPUTE EQU
COMPUTE Q
END TEST

CONDITIONS
TITLE  "Simulation {sim_num}"
OUTPUT out/sim{sim_num}
END CONDITIONS

!============================*
! 1. THERMODYNAMIC DATABASES *
!============================*
INCLUDE dtb\\elements.dtb
INCLUDE dtb\\hkf_aqu.dtb
INCLUDE dtb\\hkf_gas.dtb
INCLUDE dtb\\hkf_min.dtb

!=======================*
! 2. SOLVENT            *
!=======================*
SOLVENT
    MODEL DH2EQ3
END SOLVENT

!=================================*
! 3. SYSTEM: Pressure, T, Fluid   *
!=================================*
SYSTEM
    TdgC  {temperature}
    Pbar  {pressure}

    H   BALANCE H+         0.0
    AL  MOLE    AlOOH(AQ)  1e-13
    FE2 MOLE    Fe+2       1e-7
    FE3 MOLE    Fe+3       1e-7
    O   MOLE    H2O        55.50775721
    Na  MOLE    NA+        0.46817
    Cl  MOLE    CL-        0.54773
    Mg  MOLE    MG+2       0.05383
    Ca  MOLE    CA+2       0.01031
    MN  MOLE    MnO(AQ)    3.64e-8
    K   MOLE    K+         0.01023
    Si  MOLE    HSiO3-     4.65E-05
    S   MOLE    SO4-2      0.02825
    C   PK      CO2(g)     0
END SYSTEM

SYSTEM.ROCK
MOLE AMORPH-SILICA    {AMORPH-SILICA}
MOLE CORUNDUM         {CORUNDUM}
MOLE FERROUS-OXIDE    {FERROUS-OXIDE}
MOLE HEMATITE         {HEMATITE}
MOLE MANGANOSITE      {MANGANOSITE}
MOLE PERICLASE        {PERICLASE}
MOLE LIME             {LIME}
MOLE SODIUM-OXIDE     {SODIUM-OXIDE}
MOLE POTASSIUM-OXIDE  {POTASSIUM-OXIDE}
END SYSTEM.ROCK

!=====================*
! 10. NUMERICAL BLOCK *
!=====================*
BOX.NUMERICS
    METHOD NEWTONPRESS
    NEWTTOLF 1.0E-6
    NEWTTOLX 1.0E-6
    MAXITER 1000
END BOX.NUMERICS

SPECIES.EXCLUDE
    O2(g)
    TALCMUSCOVITE
    KAOLINITE
    TALC
    DOLOMITE
    DOLOMITE-DIS
    DOLOMITE-ORD
    DOLOMITE-sed
END SPECIES.EXCLUDE
"""

        rng = np.random.default_rng(seed=self.seed)

        # Generate random samples from input_data
        valid_values = self.input_data.values
        random_samples = rng.choice(a=valid_values, size=num_simulation, replace=True, shuffle=True)
        random_generation = pd.DataFrame(data=random_samples, columns=self.input_data.columns)

        # Create simulation files
        for idx, row in random_generation.iterrows():
            sim_num = idx + 1
            # Fill template with row values
            sim_file_content = input_template.format(
                sim_num=sim_num,
                temperature=self.temperature,
                pressure=self.pressure,
                **row[[
                    'AMORPH-SILICA', 'CORUNDUM', 'FERROUS-OXIDE',
                    'HEMATITE', 'MANGANOSITE', 'PERICLASE', 'LIME',
                    'SODIUM-OXIDE', 'POTASSIUM-OXIDE'
                ]].to_dict()
            )

            sim_file_name = f"sim{sim_num}.inn"
            sim_file_path = os.path.join(output_dir, sim_file_name)
            with open(sim_file_path, 'w') as sim_file:
                sim_file.write(sim_file_content)

            print(f"Simulation {sim_num} has been saved at {sim_file_path}.")

        return output_dir

    def process_simulation_data(self, random_size=1000, output_dir="Simulation", file_name='Random'):
        """
        Processes simulation data, generates random samples, calculates properties, 
        and augments the carbonate dataset.

        Args:
            random_size (int): Number of random samples. Default is 1000.
            output_dir (str): Directory to save generated files. Default is "Simulation".
            file_name (str): Filename for saved .xlsx. Default is 'Random'.

        Returns:
            tuple:
                - pd.DataFrame: Updated simulation result with calculated properties.
                - pd.DataFrame: Generated random samples from input data.
        """

        if self.input_data is None:
            raise ImportError("Input data cannot be None.")
        if self.simulation_result is None:
            raise ImportError("Simulation result data cannot be None.")

        rng = np.random.default_rng(seed=self.seed)

        # Generate random samples from input_data
        valid_values = self.input_data.values
        random_samples = rng.choice(a=valid_values, size=random_size, replace=True, shuffle=True)
        random_generation = pd.DataFrame(data=random_samples, columns=self.input_data.columns)
        random_generation['Simulation Number'] = np.arange(1, len(random_generation) + 1)

        # Save the random generation to Excel
        os.makedirs(output_dir, exist_ok=True)
        random_generation.to_excel(os.path.join(output_dir, f'{file_name}_{random_size}.xlsx'), index=False)

        # Calculate "Volume Rock, m3" based on the stoichiometric multipliers
        random_generation['Volume Rock, m3'] = (
            (random_generation['AMORPH-SILICA'] * 29.00) +
            (random_generation['CORUNDUM'] * 25.58) +
            (random_generation['FERROUS-OXIDE'] * 12.00) +
            (random_generation['HEMATITE'] * 30.30) +
            (random_generation['MANGANOSITE'] * 13.22) +
            (random_generation['PERICLASE'] * 11.25) +
            (random_generation['LIME'] * 16.76) +
            (random_generation['SODIUM-OXIDE'] * 27.75) +
            (random_generation['POTASSIUM-OXIDE'] * 41.04)
        ) / 1_000_000

        # Map volumes to the simulation_result
        volume_lookup = random_generation.set_index('Simulation Number')['Volume Rock, m3'].to_dict()
        self.simulation_result['Volume Rock, m3'] = self.simulation_result['Simulation Number'].map(volume_lookup)

        # Generate Sw values (log-normal distribution)
        min_value_sw = self.sw_min * 100
        max_value_sw = self.sw_max * 100
        num_samples = len(self.simulation_result)

        mu_sw = np.log((min_value_sw * max_value_sw) ** 0.5)
        sigma_sw = (np.log(max_value_sw) - np.log(min_value_sw)) / 5
        log_normal_values_sw = np.random.lognormal(mean=mu_sw, sigma=sigma_sw, size=num_samples)
        log_normal_values_sw = np.clip(log_normal_values_sw, min_value_sw, max_value_sw)

        # Generate porosity values (lognormal distribution)
        mean_porosity = self.porosity * 100
        std_dev_porosity = 0.064 * 100
        mean_log = np.log(mean_porosity / np.sqrt(1 + (std_dev_porosity / mean_porosity)**2))
        std_log = np.sqrt(np.log(1 + (std_dev_porosity / mean_porosity)**2))
        porosity_values = np.random.lognormal(mean_log, std_log, num_samples)
        porosity_values = np.clip(porosity_values, 2, 20)

        # Assign porosity and Sw to simulation_result
        self.simulation_result['Porosity'] = np.random.choice(porosity_values, size=num_samples, replace=False) / 100
        self.simulation_result['Sw'] = np.random.choice(log_normal_values_sw, size=num_samples, replace=False) / 100

        # Calculate "kg CO2"
        if self.include_dolo:
            self.simulation_result['kg CO2'] = (
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['CALCITE']) +
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['DOLOMITE-ORD'] * 2) +
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['RHODOCHROSITE']) +
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['MAGNESITE'])
            ) / 1000
        else:
            self.simulation_result['kg CO2'] = (
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['CALCITE']) +
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['RHODOCHROSITE']) +
                (self.CO2_MOLECULAR_WEIGHT * self.simulation_result['MAGNESITE'])
            ) / 1000

        # Calculate "Storage, kg CO2/m3"
        self.simulation_result['Storage, kg CO2/m3'] = (
            (self.simulation_result['kg CO2'] / self.simulation_result['Volume Rock, m3']) *
            (1 - self.simulation_result['Porosity']) *
            (1 - self.simulation_result['Sw'])
        )

        # Determine if volume ratio is above or below unity
        self.simulation_result['Above_Line'] = (
            self.simulation_result['Volume Rock, m3'] > self.simulation_result['Final Volume Rock, m3']
        )
        self.simulation_result['Above_Line_Label'] = self.simulation_result['Above_Line'].map(
            {True: 'Ratio > 1', False: 'Ratio < 1'}
        )

        self.calculate_co2 = True
        return self.simulation_result, random_generation

    def plot_ratio_comparison(self, output_dir='Simulation'):
        """
        Create a dissolution vs precipitation comparison plot.

        Args:
            output_dir (str): Directory to save the plot.
        """

        if not self.calculate_co2:
            raise PermissionError("Please process the data first (call process_simulation_data).")

        palette = {'Ratio > 1': 'red', 'Ratio < 1': 'green'}

        # Determine plotting range
        min_val = min(
            self.simulation_result['Final Volume Rock, m3'].min(),
            self.simulation_result['Volume Rock, m3'].min()
        )
        max_val = max(
            self.simulation_result['Final Volume Rock, m3'].max(),
            self.simulation_result['Volume Rock, m3'].max()
        )
        buffer = (max_val - min_val) * 0.05
        min_val -= buffer
        max_val += buffer

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=self.simulation_result,
            x='Final Volume Rock, m3',
            y='Volume Rock, m3',
            hue='Above_Line_Label',
            palette=palette
        )

        # 45-degree reference line
        x_vals = np.linspace(min_val, max_val, 100)
        plt.plot(x_vals, x_vals, color='black', linestyle='--', label='Ratio = 1')

        # Labeling, grid, style
        plt.xlabel('Final Precipitated Volume (m³)')
        plt.ylabel('Initial Mineral Volume (m³)')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.annotate(
            r'$\text{Ratio} = \frac{\text{Initial}}{\text{Final}}$',
            xy=(min_val + 0.2 * (max_val - min_val), max_val - 0.1 * (max_val - min_val)),
            fontsize=14,
            bbox=dict(boxstyle="round", fc="white", ec="black")
        )

        plt.legend(loc='upper left')
        plt.grid(alpha=0.5)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        if self.include_dolo:
            plt.savefig(os.path.join(output_dir, 'Ratio_Plot_with_Dolomite.pdf'), format='pdf')
        else:
            plt.savefig(os.path.join(output_dir, 'Ratio_Plot_without_Dolomite.pdf'), format='pdf')
        plt.show()

    def MonteCarloStorage(self, num_simulations=10000, output_dir='Simulation'):
        """
        Simulate random storage scenarios and plot the Monte-Carlo with P10, P50, and P90.

        Args:
            num_simulations (int): Number of random simulations. Default is 10000.
            output_dir (str): Directory to save the plot.
        """

        if not self.calculate_co2:
            raise PermissionError("Please process the data first (call process_simulation_data).")

        rng = np.random.default_rng(seed=self.seed)

        valid_co2 = self.simulation_result['kg CO2'].values
        valid_vol = self.simulation_result['Volume Rock, m3'].values
        valid_porosity = self.simulation_result['Porosity'].values
        valid_sw = self.simulation_result['Sw'].values

        random_co2 = rng.choice(a=valid_co2, size=num_simulations, replace=True, shuffle=True)
        random_vol = rng.choice(a=valid_vol, size=num_simulations, replace=True, shuffle=True)
        random_por = rng.choice(a=valid_porosity, size=num_simulations, replace=True, shuffle=True)
        random_sw = rng.choice(a=valid_sw, size=num_simulations, replace=True, shuffle=True)

        # Compute storage and gross rock volume
        storage_data = []
        for co2_val, vol_val, phi_val, sw_val in zip(random_co2, random_vol, random_por, random_sw):
            storage = (co2_val / vol_val) * (1 - phi_val) * (1 - sw_val)
            grv = self.target / storage
            storage_data.append((storage, grv, phi_val, sw_val))

        # Sort by storage capacity descending
        sorted_data = sorted(storage_data, key=lambda x: x[0], reverse=True)
        x1 = [d[0] for d in sorted_data]
        y1 = np.arange(len(x1)) / float(len(x1))

        # Interpolation for P10, P50, P90
        inter_storage = interpolate.splrep(y1, x1)
        p10_storage = interpolate.splev(0.1, inter_storage)
        p50_storage = interpolate.splev(0.5, inter_storage)
        p90_storage = interpolate.splev(0.9, inter_storage)

        # Retrieve the actual data for P90, P50, P10
        p90_data = next(d for d in sorted_data if d[0] <= p90_storage)
        p50_data = next(d for d in sorted_data if d[0] <= p50_storage)
        p10_data = next(d for d in sorted_data if d[0] <= p10_storage)

        p90_volume, p90_poro, p90_sw = p90_data[1], p90_data[2], p90_data[3]
        p50_volume, p50_poro, p50_sw = p50_data[1], p50_data[2], p50_data[3]
        p10_volume, p10_poro, p10_sw = p10_data[1], p10_data[2], p10_data[3]

        plt.figure(figsize=(12, 8))
        sns.set_palette('husl', 8)

        # Plot ECDF (Complementary)
        storage_plot = sns.ecdfplot(x1, complementary=True, color='orange')

        # Prepare annotation box style
        bbox = dict(boxstyle="round", fc="0.8", edgecolor="black")
        arrowprops = dict(arrowstyle="->", color="black", lw=1)

        # Format for volumes
        volume_format = "{:.2e}"

        # Annotate P90
        plt.annotate(
            '   $P_{90}$\n$C_{{CO_2}} = {:.0f}$ kg CO₂/m³\nϕ = {:.2f}%\n$S_w$ = {:.2f}%\nGRV = {} m³'.format(
                p90_storage, p90_poro * 100, p90_sw * 100, volume_format.format(p90_volume)
            ),
            xy=(p90_storage, 0.9), xytext=(p90_storage + 10, 0.85),
            bbox=bbox, arrowprops=arrowprops
        )

        # Annotate P50
        plt.annotate(
            '   $P_{50}$\n$C_{{CO_2}} = {:.0f}$ kg CO₂/m³\nϕ = {:.2f}%\n$S_w$ = {:.2f}%\nGRV = {} m³'.format(
                p50_storage, p50_poro * 100, p50_sw * 100, volume_format.format(p50_volume)
            ),
            xy=(p50_storage, 0.5), xytext=(p50_storage + 10, 0.6),
            bbox=bbox, arrowprops=arrowprops
        )

        # Annotate P10
        plt.annotate(
            '   $P_{10}$\n$C_{{CO_2}} = {:.0f}$ kg CO₂/m³\nϕ = {:.2f}%\n$S_w$ = {:.2f}%\nGRV = {} m³'.format(
                p10_storage, p10_poro * 100, p10_sw * 100, volume_format.format(p10_volume)
            ),
            xy=(p10_storage, 0.1), xytext=(p10_storage + 10, 0.2),
            bbox=bbox, arrowprops=arrowprops
        )

        # Vertical & horizontal lines for P10, P50, P90
        storage_plot.vlines(
            x=[p90_storage, p50_storage, p10_storage],
            ymin=[0, 0, 0],
            ymax=[0.9, 0.5, 0.1],
            colors=['red', 'blue', 'green'],
            linestyles='--'
        )
        storage_plot.hlines(
            y=[0.9, 0.5, 0.1],
            xmin=[0, 0, 0],
            xmax=[p90_storage, p50_storage, p10_storage],
            colors=['red', 'blue', 'green'],
            linestyles='--'
        )

        plt.xlabel(r'Storage Capacity (kg CO₂/m³)')
        plt.ylabel('Probability')
        plt.xlim(min(x1), max(x1))
        plt.grid(alpha=0.5)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)

        if self.include_dolo:
            plt.savefig(os.path.join(output_dir, 'Monte_Carlo_with_Dolomite.pdf'), format='pdf')
        else:
            plt.savefig(os.path.join(output_dir, 'Monte_Carlo_without_Dolomite.pdf'), format='pdf')

        plt.show()


class GasProperties(IFPMin):
    """
    Inherits from IFPMin and adds methods for CO2 PVT calculations using CoolProp.
    """

    def __init__(
        self,
        input_data=None,
        simulation_result=None,
        target=30e9,
        include_dolo=True,
        temperature=None,
        pressure=None,
        porosity=None,
        porosity_max=None,
        sw_min=None,
        sw_max=None,
        seed=141
    ):
        super().__init__(
            input_data, simulation_result, target, include_dolo,
            temperature, pressure, porosity, porosity_max, sw_min, sw_max, seed
        )

        self.T_K = self.temperature + 273.15  # Kelvin
        self.P_pa = self.pressure * 1e5       # Pascal

    def co2_density(self):
        """
        Return CO2 density (kg/m³) at (P, T) via Helmholtz energy EoS (CoolProp).
        """

        rho = CP.PropsSI('D', 'P', self.P_pa, 'T', self.T_K, 'CO2')

        return rho

    def co2_viscosity(self):
        """
        Return CO2 viscosity (cP) at (P, T) via Helmholtz energy EoS (CoolProp).
        """

        mu_pa_s = CP.PropsSI('V', 'P', self.P_pa, 'T', self.T_K, 'CO2')  # Pa·s

        mu_cp = mu_pa_s * 1000.0  # convert Pa·s to cP

        return mu_cp

    def co2_z_factor(self):
        """
        Return CO2 Z-factor (dimensionless) at (P, T).
        """

        Z = CP.PropsSI('Z', 'P', self.P_pa, 'T', self.T_K, 'CO2')

        return Z

    def co2_volume_factor(self):
        """
        Return formation volume factor (Bg) of CO2 (dimensionless).
        """

        # Standard conditions in the denominator: P = 101325 Pa, T = 298.15 K
        Bg = (101325 / 298.15) * (self.co2_z_factor() * self.T_K) / self.P_pa

        return Bg

    def real_gas_pseudo_pressure(self, Pinj):
        """
        Compute the real-gas pseudo-pressure from Pinj to self.pressure:
            m(p) = ∫(2 * p / (mu_g * z)) dp

        Args:
            Pinj (float): Injection pressure in bar.

        Returns:
            float: The definite integral of the real-gas pseudo-pressure.
        """

        Pinj_psi = Pinj * 14.5038  # Pinj bar → psi
        Pres = self.pressure * 14.5038 # Pres bar → psi

        def integrand(p):
            mu = self.co2_viscosity()
            z_val = self.co2_z_factor()
            return 2.0 * p / (mu * z_val)

        result, _ = quad(integrand, Pinj_psi, Pres, limit=1000, epsabs=1.49e-4, epsrel=1.49e-4)
        return round(result, 2)


class IFlowP(GasProperties):
    """
    Extension of GasProperties to handle injection-pressure related calculations.
    """

    def __init__(
        self,
        input_data=None,
        simulation_result=None,
        target=30e9,
        include_dolo=True,
        temperature=None,
        pressure=None,
        porosity=None,
        porosity_max=None,
        sw_min=None,
        sw_max=None,
        seed=141,
        Pinj=None
    ):
        super().__init__(
            input_data, simulation_result, target, include_dolo,
            temperature, pressure, porosity, porosity_max, sw_min, sw_max, seed
        )
        if Pinj is None:
            raise ValueError("Please define your Pinj (injection pressure)!")
        self.Pinj = Pinj  # psia or bar, depending on usage

    def update_pinj(self, new_Pinj):
        """
        Update the Injection Pressure (Pinj).

        Args:
            new_Pinj (float): The new injection pressure value.
        """

        if new_Pinj < 0:
            raise ValueError("Injection pressure must be positive.")
        
        self.Pinj = new_Pinj

    def co2_injectivity(self, k, h, rw, skin):
        """
        Compute single-phase gas injection rate.

        Args:
            k (float): Permeability in mD.
            h (float): Reservoir thickness in meters.
            rw (float): Wellbore radius in meters.
            skin (float): Dimensionless skin factor.

        Returns:
            J: Approximate injectivity index (tonnes CO₂/${bar^2}$.cp.year).
        """
        
        mu_g = self.co2_viscosity()
        gamma = 1.8021 / 1.293  # Specific gravity of CO2 with respect to air

        # Pseudo-pressure difference from Pinj to reservoir pressure
        delta_psi = abs(self.real_gas_pseudo_pressure(Pinj=self.Pinj))

        # Convert to field units
        T_rankine = (9.0 / 5.0) * self.temperature + 491.67  # °R
        h_ft = 3.28084 * h # ft
        rw_ft = 3.28084 * rw # ft
        kh = k * h_ft

        # Beta correlation from Eq. 7.116b
        beta = (4.85e4) / ((self.porosity ** 5.5) * (k ** 0.5))

        # Non-Darcy/turbulent coefficient from Eq. 7.138
        D = ((2.22e-15 * gamma) / (mu_g * rw_ft * h_ft)) * beta * k

        # Quadratic eq: Q^2 * b2 + Q * a2 - delta_psi = 0
        a2 = ((1422.0 * T_rankine) / kh) * (7 + skin)
        b2 = ((1422.0 * T_rankine) / kh) * D
        inside_sqrt = a2**2 + (4.0 * b2 * delta_psi)

        if inside_sqrt < 0:
            raise ValueError("Negative discriminant => no real solution for Qg.")

        Qg_turb = (-a2 + math.sqrt(inside_sqrt)) / (2.0 * b2)

        if Qg_turb < 0:
            raise ValueError(f"Solved Qg <= 0 => non-physical. Qg={Qg_turb}")


        delta_p = delta_psi * 0.0689476 * 0.0689476  # bar^2.cp
        Qg = (28.316846592 * Qg_turb * 1.8021 / 1000) * 365  # tonnes/year

        # Approx. injectivity index
        J = Qg / math.sqrt(delta_p)

        border_char = "|"
        padding = 4
        line_width = 44
        inner_width = line_width - (2 * padding) - 2

        print("=" * line_width)
        print(f"{border_char}{' ' * padding}{'CO2 Analytical Simulation':^{inner_width}}{' ' * padding}{border_char}")
        print("=" * line_width)
        print(f"{border_char}{' ' * padding}{'Results':^{inner_width}}{' ' * padding}{border_char}")
        print("=" * line_width)
        print(f"{border_char}{' ' * padding}{f'Mechanical Skin     =':<25}{skin:>{inner_width - 25}.2f}{' ' * padding}{border_char}")
        print(f"{border_char}{' ' * padding}{f'Rate Dependent Skin =':<25}{round(D * Qg_turb, 2):>{inner_width - 25}.2f}{' ' * padding}{border_char}")
        print(f"{border_char}{' ' * padding}{f'Effective Skin      =':<25}{round(skin + D * Qg_turb, 2):>{inner_width - 25}.2f}{' ' * padding}{border_char}")
        print(f"{border_char}{' ' * padding}{f'Injectivity Index   =':<25}{round(J, 2):>{inner_width - 25}.2f}{' ' * padding}{border_char}")
        print("=" * line_width)

        return round(J, 2)
