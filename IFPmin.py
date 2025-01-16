import os
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class IFPMin:
    CO2_MOLECULAR_WEIGHT = 44.01  # g/mol
    DEFAULT_TEMPERATURE = 40.0    # degrees Celsius
    DEFAULT_PRESSURE = 249        # bar
    DEFAULT_POROSITY = 0.089      # fraction
    DEFAULT_POROSITY_MAX = 0.2    # fraction
    DEFAULT_SW_MIN = 0.30         # fraction
    DEFAULT_SW_MAX = 0.95         # fraction

    def __init__(self, input_data=None, simulation_result=None, target=30 * 10**9, 
                 include_dolo=True, temperature=None, pressure=None, porosity=None, porosity_max=None, sw_min=None, sw_max=None, seed=141):
        """
        Initialize the CO2 storage capacity calculator.

        Args:
            input_data (DataFrame): A pandas DataFrame containing the input minerals.
            simulation_result (DataFrame): A pandas DataFrame containing the precipitated minerals.
            target (float): Storage target of CO2 (kg).
            include_dolo (bool): Whether to include mineral dolomite during the calculations.
            seed (int): Random seed for resampling.
            temperature (float, optional): Temperature for the simulation (°C). Default is 40.0.
            pressure (float, optional): Pressure for the simulation (bar). Default is 249.
        """
        
        self.input_data = input_data
        self.simulation_result = simulation_result
        self.target = target
        self.include_dolo = include_dolo
        self.seed = seed
        np.random.seed(self.seed)

        # Allow overriding default temperature and pressure
        self.temperature = temperature if temperature is not None else self.DEFAULT_TEMPERATURE
        self.pressure = pressure if pressure is not None else self.DEFAULT_PRESSURE
        self.porosity = porosity if porosity is not None else self.DEFAULT_POROSITY
        self.porosity_max = porosity_max if porosity_max is not None else self.DEFAULT_POROSITY
        self.sw_min = sw_min if sw_min is not None else self.DEFAULT_SW_MIN
        self.sw_max = sw_max if sw_max is not None else self.DEFAULT_SW_MAX



    def generate_simulation_files(self, output_dir="Simulation", num_simulation=1000):
        """
        Generate simulation input files from a dataset.

        Args:
            output_dir (str): The directory to save the simulation input files.
        
        Returns:
            .inn files: Simulation input files for the CO2 mineralization (ArXim).
        """

        if self.input_data is None:
            raise ImportError("Input file cannot be imported!")
        
        if self.simulation_result is None:
            raise ImportError("Simulation file cannot be imported!")
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define a template for the input file with placeholders for mole values
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
            MODEL DH2EQ3         ! BDot model for moderate to slightly high ionic strengths
        END SOLVENT
        
        !=================================*
        ! 3. SYSTEM: Pressure, T, Fluid   *
        !=================================*
        SYSTEM
            TdgC  {temperature}
            Pbar  {pressure}
        
        ! Baseline formation water:
            H   BALANCE H+         0.0            ! pH as charge balance
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
            C   PK      CO2(g)     0 !CO2(aq) in equilibrium with CO2(g), corresponding to the total pressure.
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
        
        ! (Optional) Exclude or include species here
        SPECIES.EXCLUDE
            O2(g)  ! For instance, ignoring free O2 gas
            TALCMUSCOVITE
            KAOLINITE
            !QUARTZ-A
            !MUSCOVITE
            TALC
            DOLOMITE
            DOLOMITE-DIS
            DOLOMITE-ORD
            DOLOMITE-sed
        END SPECIES.EXCLUDE
        """

        rng = np.random.default_rng(seed=self.seed)
        
        # Generate random samples from mole data
        valid_values = self.input_data.values
        random = rng.choice(a=valid_values, size=num_simulation, replace=True, shuffle=True)
        random_generation = pd.DataFrame(data=random, columns=self.input_data.columns)

        # Iterate through the dataset and create input files
        for idx, row in random_generation.iterrows():
            sim_num = idx + 1  # Simulation number starts at 1
            sim_file_content = input_template.format(
                sim_num=sim_num,
                temperature=self.temperature,
                pressure=self.pressure,
                **row[['AMORPH-SILICA', 'CORUNDUM', 'FERROUS-OXIDE',
                    'HEMATITE', 'MANGANOSITE', 'PERICLASE', 'LIME',
                    'SODIUM-OXIDE', 'POTASSIUM-OXIDE']].to_dict()
            )
            
            # Define the file name and save the content
            sim_file_name = f"sim{sim_num}.inn"
            sim_file_path = os.path.join(output_dir, sim_file_name)
            with open(sim_file_path, 'w') as sim_file:
                sim_file.write(sim_file_content)
                logging.basicConfig(level=logging.INFO)
                logging.info(f"Simulation {sim_num} has been saved at {sim_file_path}.")

        return output_dir  # Return the directory where the files are saved

    def process_simulation_data(self, random_size=1000, output_dir="Simulation", file_name='Random'):
        """
        Processes simulation data, generates random samples, calculates properties, and augments the carbonate dataset.
        
        Args:
            random_size (int): Number of random sampling. Default is 1000.
            output_dir (str): Directory for saving generated files. Default is "Simulation".
            file_name (str): File name for saving generated files. Default is "Random".
        
        Returns:
            DataFrame: Updated simulation result dataset with additional calculated properties.
        """

        rng = np.random.default_rng(seed=self.seed)
        
        # Generate random samples from mole data
        valid_values = self.input_data.values
        random = rng.choice(a=valid_values, size=random_size, replace=True, shuffle=True)
        random_generation = pd.DataFrame(data=random, columns=self.input_data.columns)
        random_generation['Simulation Number'] = [i + 1 for i in range(len(random_generation))]
        
        # Save the generated data to an Excel file
        os.makedirs(output_dir, exist_ok=True)
        random_generation.to_excel(os.path.join(output_dir, f'{file_name}_{random_size}.xlsx'), index=False)
        
        # Calculate 'Volume Rock, m3'
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
        

        # Map 'Volume Rock, m3' to the carbonate dataset
        volume_lookup = random_generation.set_index('Simulation Number')['Volume Rock, m3'].to_dict()
        self.simulation_result['Volume Rock, m3'] = self.simulation_result['Simulation Number'].map(volume_lookup)
        
        # Generate Sw values using a log-normal distribution
        min_value_sw = self.sw_min * 100
        max_value_sw = self.sw_max  * 100

        num_samples = len(self.simulation_result)
        mu_sw = np.log((min_value_sw * max_value_sw) ** 0.5)
        sigma_sw = (np.log(max_value_sw) - np.log(min_value_sw)) / 5
        log_normal_values_sw = np.random.lognormal(mean=mu_sw, sigma=sigma_sw, size=num_samples)
        log_normal_values_sw = np.clip(log_normal_values_sw, min_value_sw, max_value_sw)
        
        # Generate porosity values using a lognormal distribution
        mean_porosity = self.porosity * 100  # Mean porosity (percent)
        std_dev_porosity = 0.064 * 100  # Standard deviation (percent)
        mean_log = np.log(mean_porosity / np.sqrt(1 + (std_dev_porosity / mean_porosity)**2))
        std_log = np.sqrt(np.log(1 + (std_dev_porosity / mean_porosity)**2))
        porosity_values = np.random.lognormal(mean_log, std_log, num_samples)
        porosity_values = np.clip(porosity_values, 2, 20)
        
        # Assign porosity and Sw to the dataset
        self.simulation_result['Porosity'] = np.random.choice(porosity_values, size=num_samples, replace=False) / 100
        self.simulation_result['Sw'] = np.random.choice(log_normal_values_sw, size=num_samples, replace=False) / 100
        
        # Calculate 'kg CO2'
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
        
        # Calculate 'Storage, kg CO2/m3'
        self.simulation_result['Storage, kg CO2/m3'] = (
            (self.simulation_result['kg CO2'] / self.simulation_result['Volume Rock, m3']) *
            (1 - self.simulation_result['Porosity']) *
            (1 - self.simulation_result['Sw'])
        )
        
        # Add 'Above_Line' and 'Above_Line_Label'
        self.simulation_result['Above_Line'] = self.simulation_result['Volume Rock, m3'] > self.simulation_result['Final Volume Rock, m3']
        self.simulation_result['Above_Line_Label'] = self.simulation_result['Above_Line'].map({True: 'Ratio > 1', False: 'Ratio < 1'})

        self.calculate_co2 = True
        
        return self.simulation_result, random_generation

    def plot_ratio_comparison(self, output_dir='Simulation'):
        """
        Creates a scatter plot comparing the initial and final volumes, with annotations and a 45-degree line.

        Args:
            output_dir (str): Directory for saving generated files. Default is "Simulation".
        """

        if not self.calculate_co2: 
            raise PermissionError("Please process the data first!")
        
        # Define color palette
        palette = {'Ratio > 1': 'red', 'Ratio < 1': 'green'}

        # Determine the range dynamically based on the dataset
        min_val = min(self.simulation_result['Final Volume Rock, m3'].min(), self.simulation_result['Volume Rock, m3'].min())
        max_val = max(self.simulation_result['Final Volume Rock, m3'].max(), self.simulation_result['Volume Rock, m3'].max())

        # Slightly extend the range for better visualization
        buffer = (max_val - min_val) * 0.05  # 5% buffer
        min_val -= buffer
        max_val += buffer

        # Set figure size
        plt.figure(figsize=(12, 8))

        # Scatter plot with conditional coloring
        sns.scatterplot(
            data=self.simulation_result,
            x='Final Volume Rock, m3',
            y='Volume Rock, m3',
            hue='Above_Line_Label',
            palette=palette
        )

        # Define the range for the 45-degree line (y = x)
        x_vals = np.linspace(min_val, max_val, 100)
        y_vals = x_vals  # For y = x, the y-values are the same as x-values

        # Add the 45-degree line
        plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Ratio = 1')

        # Labels and limits
        plt.xlabel('Final Precipitated Volume, $\mathrm{m}^3$')
        plt.ylabel('Initial Mineral Volume, $\mathrm{m}^3$')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # Add annotation
        plt.annotate(
            r'$\text{Ratio} = \frac{\text{Initial Volume}}{\text{Final Volume}}$',  # LaTeX-style equation
            xy=(min_val + 0.2 * (max_val - min_val), max_val - 0.1 * (max_val - min_val)),  # Annotation position
            xycoords='data',
            fontsize=20,
            bbox=dict(boxstyle="round", fc="white", ec="black")  # Box style for annotation
        )

        # Legend and title
        plt.legend(loc='upper left')

        # Set font properties
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 16

        # Add grid and layout adjustments
        plt.grid(alpha=0.5)
        plt.tight_layout()

        # Save plot based on the dolo flag
        if self.include_dolo:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/Ratio_Plot_with_Dolomite.pdf', format='pdf')
        
        else:
            plt.savefig(f'{output_dir}/Ratio_Plot_without_Dolomite.pdf', format='pdf')

        # Show plot
        plt.show()

    def MonteCarloStorage(self, num_simulations=10000, output_dir='Simulation'):
        """
        Simulates random storage scenarios and plots the ECDF with P10, P50, P90 annotations.

        Args:
            num_simulations (int): Number of random simulations to perform. Default is 10000.
            output_dir (str): Directory for saving generated files. Default is "Simulation".
        """

        rng_volume = np.random.default_rng(seed=self.seed)

        # Extract valid values
        valid_values_co2 = self.simulation_result['kg CO2'].values
        valid_values_m3 = self.simulation_result['Volume Rock, m3'].values
        valid_values_poro = self.simulation_result['Porosity'].values
        valid_values_sw = self.simulation_result['Sw'].values

        # Generate random values
        random_co2 = rng_volume.choice(a=valid_values_co2, size=num_simulations, replace=True, shuffle=True)
        random_m3 = rng_volume.choice(a=valid_values_m3, size=num_simulations, replace=True, shuffle=True)
        random_poro = rng_volume.choice(a=valid_values_poro, size=num_simulations, replace=True, shuffle=True)
        random_sw = rng_volume.choice(a=valid_values_sw, size=num_simulations, replace=True, shuffle=True)

        # Calculate storage and gross rock volume
        storage_list = []
        storage_data = []

        for co2, vol, phi, sw in zip(random_co2, random_m3, random_poro, random_sw):
            storage = (co2 / vol) * (1 - phi) * (1 - sw)
            grv = self.target / storage
            storage_list.append(storage)
            storage_data.append((storage, grv, phi, sw))

        # Sort storage data
        sorted_storage_data = sorted(storage_data, key=lambda x: x[0], reverse=True)
        x1 = [data[0] for data in sorted_storage_data]  # Sorted storage values
        y1 = np.arange(len(x1)) / float(len(x1))

        # Interpolate for P10, P50, P90
        inter_storage = interpolate.splrep(y1, x1)
        p10_storage = interpolate.splev(0.1, inter_storage)
        p50_storage = interpolate.splev(0.5, inter_storage)
        p90_storage = interpolate.splev(0.9, inter_storage)

        # Extract corresponding parameters
        p90_data = next(data for data in sorted_storage_data if data[0] <= p90_storage)
        p50_data = next(data for data in sorted_storage_data if data[0] <= p50_storage)
        p10_data = next(data for data in sorted_storage_data if data[0] <= p10_storage)

        # Unpack data
        p90_volume, p90_poro, p90_sw = p90_data[1], p90_data[2], p90_data[3]
        p50_volume, p50_poro, p50_sw = p50_data[1], p50_data[2], p50_data[3]
        p10_volume, p10_poro, p10_sw = p10_data[1], p10_data[2], p10_data[3]

        # Plot storage capacity ECDF
        plt.figure(figsize=(12, 8))
        sns.set_palette('husl', 8)

        storage_plot = sns.ecdfplot(storage_list, complementary=True, color='orange')

        bbox = dict(boxstyle="round", fc="0.8", edgecolor="black")
        arrowprops = dict(arrowstyle="->", color="black", lw=1)
        volume_format = "{:.2e}"

        r'$P_{90}$' '\n'
        r'$C_{CO_2} = {:.0f}$ kg $\mathrm{CO_2}/\mathrm{m}^3$' '\n'
        r'$\phi = {:.2f}\%$' '\n'
        r'$S_w = {:.2f}\%$' '\n'
        r'$GRV = {:.2e} \, \mathrm{m}^3$'

        plt.annotate(
            '              $\mathrm{P_{90}}$\n$\mathrm{C_{CO_2}}$ = %d kg $\mathrm{CO_2}/\mathrm{m}^3$\nϕ = %.2f%%\n$\mathrm{S_w}$ = %.2f%%\nGRV = %s $\mathrm{m}^3$' % 
            (p90_storage, p90_poro * 100, p90_sw * 100, volume_format.format(p90_volume)),
            xy=(p90_storage, 0.9), xytext=(p90_storage + 10, 0.85),
            bbox=bbox, arrowprops=arrowprops
        )

        # Annotations for P50
        plt.annotate(
            '              $\mathrm{P_{50}}$\n$\mathrm{C_{CO_2}}$ = %d kg $\mathrm{CO_2}/\mathrm{m}^3$\nϕ = %.2f%%\n$\mathrm{S_w}$ = %.2f%%\nGRV = %s $\mathrm{m}^3$' % 
            (p50_storage, p50_poro * 100, p50_sw * 100, volume_format.format(p50_volume)),
            xy=(p50_storage, 0.5), xytext=(p50_storage + 10, 0.6),
            bbox=bbox, arrowprops=arrowprops
        )

        # Annotations for P10
        plt.annotate(
            '              $\mathrm{P_{10}}$\n$\mathrm{C_{CO_2}}$ = %d kg $\mathrm{CO_2}/\mathrm{m}^3$\nϕ = %.2f%%\n$\mathrm{S_w}$ = %.2f%%\nGRV = %s $\mathrm{m}^3$' % 
            (p10_storage, p10_poro * 100, p10_sw * 100, volume_format.format(p10_volume)),
            xy=(p10_storage, 0.1), xytext=(p10_storage + 10, 0.2),
            bbox=bbox, arrowprops=arrowprops
        )

        # Add vertical and horizontal lines
        storage_plot.vlines(x=[p90_storage, p50_storage, p10_storage], ymin=[0, 0, 0], 
                            ymax=[0.9, 0.5, 0.1], colors=['red', 'blue', 'green'], ls='--')
        storage_plot.hlines(y=[0.9, 0.5, 0.1], xmin=[0, 0, 0], xmax=[p90_storage, p50_storage, p10_storage], 
                            colors=['red', 'blue', 'green'], ls='--')

        # Add vertical and horizontal lines
        storage_plot.vlines(x=[p90_storage, p50_storage, p10_storage], ymin=[0, 0, 0], 
                            ymax=[0.9, 0.5, 0.1], colors=['red', 'blue', 'green'], ls='--')
        storage_plot.hlines(y=[0.9, 0.5, 0.1], xmin=[0, 0, 0], xmax=[p90_storage, p50_storage, p10_storage], 
                            colors=['red', 'blue', 'green'], ls='--')

        # Labels and settings
        plt.xlabel(r'Storage Capacity, kg $\mathrm{CO_2}/\mathrm{m}^3$')
        plt.ylabel('Probability')
        plt.rcParams['font.size'] = 16
        plt.rcParams['font.family'] = 'Arial'
        plt.xlim(min(storage_list), max(storage_list))

        plt.grid(alpha=0.5)
        plt.tight_layout()

        if self.include_dolo:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f'{output_dir}/Monte_Carlo_with_Dolomite.pdf', format='pdf')
        
        else:
            plt.savefig(f'{output_dir}/Monte_Carlo_without_Dolomite.pdf', format='pdf')

        plt.show()
