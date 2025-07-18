import os
import re
import logging
import subprocess
import numpy as np
import math
from scipy.stats import qmc, triang
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Configure root logger
t = logging.getLogger()
t.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
t.addHandler(handler)
# ─────────────────────────────────────────────────────────────────────────────

class CO2InjectionConfig:
    """
    Class to read, validate, and run CO2 injection simulation workflows.
    """
    SECTION_REGEX = re.compile(r'^(\w+)(\s*/)?$')

    def __init__(self):
        self.seed = 42
        self.config = {}

        banner = r"""
=========================================================================
  _____    _____   _____   __  __   _____    _   ___ 
 |_   _|  |  ___| |  __ \ |  \/  | |_   _|  | \ |   |
   | |    | |_    | |__) || |\/| |   | |    |  \|   |
   | |    |  _|   |  ___/ | |  | |   | |    |  |\   |
  _| |_   | |     | |     | |  | |  _| |_   |  | \  |
 |_____|  |_|     |_|     |_|  |_| |_____|  |__|  \_|

=========================================================================               
IFP School mineralization software
Developed by:
• Berat Tuğberk Günel
• Mahmoud Khaled

Laurie Dake Challenge 2025

--> The usage of this software does not seek any financial purposes
=========================================================================
"""
        t.info(banner)

        # read Input.data from cwd
        self._read_input_file()

        # retrieve paths parsed in SIMTYPE PATH block
        paths = self.config.get('paths', {})
        root = os.path.dirname(os.path.abspath(self.input_dir))

        self.output_dir = os.path.join(root, 'out')
        arxim_exe = paths.get('arxim_exe', '')
        self.arxim_exe = os.path.join(root, arxim_exe)

        self.arxim_dir = os.path.dirname(os.path.abspath(self.arxim_exe))

        sim = self.config.get('simtype', {})

        # 1) CRTSIM: parse mineralogy & generate .inn files
        if sim.get('CRTSIM'):
            t.info('CRTSIM: Validating prerequisites...')
            obs, rp = self._validate_crtsim()

            t.info(f"CRTSIM: Prerequisities validates successfully. Running...\n obs={obs - 1} | P={rp['pressure_bar']} bar | T={rp['temperature_C']} °C")

            t.info('CRTSIM: Reading Minerology.min file...')

            self._read_mineralogy()

            t.info('CRTSIM: Generating the realizations...')
            self._generate_simulation_files(
                minerology_df = pd.DataFrame(self.config['mineralogy']['observations']),
                res_pres      = self.config['resprop']['pressure_bar'],
                res_temp      = self.config['resprop']['temperature_C'],
                include_dolo  = sim.get('TDYNSIM', False)
            )
            t.info('CRTSIM: Simulation successful!')

        # 2) TDYNSIM: validate & run ArXim
        if sim.get('TDYNSIM'):
            t.info('TDYNSIM: Validating prerequisites...')
            self._validate_tdynsim()

            t.info('TDYNSIM: Prerequisites validated successfully. Running......')
            self._run_thermodynamic_simulations()

            # 3) Extract, count & analyze equilibrium species
            t.info('TDYNSIM: Extracting equilibrium species from ArXim out/')
            self._extract_equilibrium_species()

            t.info('TDYNSIM: Counting successful simulations...')
            self._count_successful_simulations()

            t.info('TDYNSIM: Analyzing and exporting results...')
            self._analyze_results()

            t.info('TDYNSIM: Simulation successful!')
        
        # STGSIM: compute storage capacity
        if sim.get('STGSIM'):
            t.info('STGSIM: Validating prerequisites...')

            self._validate_stgsim()

            t.info('STGSIM: Prerequisites validated successfully. Running......')
            t.info('STGSIM: Processing simulation data for storage capacity estimation...')

            df = self._process_simulation_data()
            self._plot_ratio_comparison(df=df)
            self._SimulateStorageCapacity(df=df)

            t.info('STGSIM: Simulation successful!')

        # INJSIM: compute the injectivity
        if sim.get('INJSIM'):
            t.info('INJSIM: validating prerequisites...')
            params = self._validate_injsim()
            t.info(f"INJSIM: prerequisites validated successfully. Running...\n INJSIM: A={params['A']}, Fb={params['Fb']}")

            p50_time, p50_press = self._SimulateInjectivity()
            self._plot_pressure(time=p50_time, pressure=p50_press)

            t.info('INJSIM: Simulation successful!')
        
        # Final validation
        self._validate()
        t.info('SYSTEM: All simulation workflows completed!')

    def _read_input_file(self):
        # Locate Input.data anywhere under the working directory
        root = os.getcwd()
        input_data_path = None
        for dirpath, _, filenames in os.walk(root):
            if 'Input.data' in filenames:
                input_data_path = os.path.join(dirpath, 'Input.data')
                self.input_dir = dirpath

                break

        if not input_data_path:
            t.error(f"SYSTEM: Input.data not found under {root}")
            sys.exit("SYSTEM: Aborting IFPmin...")

        t.info(f"SYSTEM: Reading Input.data file at {input_data_path}...")

        with open(input_data_path, encoding='utf-8-sig') as f:
            lines = [ln.split('--',1)[0].strip() for ln in f if ln.strip()]
        idx = 0

        while idx < len(lines):
            m = self.SECTION_REGEX.match(lines[idx])
            if m and not m.group(2):
                sec = m.group(1).upper()
                if sec == 'TARGET':
                    idx = self._parse_target(lines, idx+1)
                    continue
                if sec == 'SIMTYPE':
                    idx = self._parse_simtype(lines, idx+1)
                    continue
                if sec == 'RESPROP':
                    idx = self._parse_resprop(lines, idx+1)
                    continue
                if sec == 'WELLSPEC':
                    idx = self._parse_wellspec(lines, idx+1)
                    continue
            idx += 1

    def _parse_target(self, lines, start):
        p = lines[start].split()
        self.config['target'] = {
            'style': p[0],
            'storage_objective_t_per_year': float(p[1]),
            'time_start_yr':                float(p[2]),
            'time_end_yr':                  float(p[3])
        }
        for i in range(start, len(lines)):
            if lines[i].upper().startswith('TARGET') and '/' in lines[i]:
                return i+1
        
        return len(lines)

    def _parse_simtype(self, lines, start):
        sim, order = {}, []
        injsim_params = None
        i = start

        while i < len(lines):
            line = lines[i].strip()
            # end of SIMTYPE block
            if line.upper().startswith('SIMTYPE') and '/' in line:
                break

            # PATH block: parse input and arxim paths
            if line.upper() == 'PATH':
                j = i + 1
                while j < len(lines) and not (lines[j].upper().startswith('PATH') and '/' in lines[j]):
                    parts = lines[j].split()
                    if len(parts) >= 2:
                        self.config.setdefault('paths', {})['input_dir'] = parts[0]
                        self.config['paths']['arxim_exe'] = parts[1]
                    j += 1
                i = j + 1
                continue

            parts = line.split()
            if parts:
                key = parts[0].upper()
                order.append(key)
                if key == 'INJSIM' and len(parts) == 3:
                    sim[key] = True
                    try:
                        A, Fb = float(parts[1]), float(parts[2])
                    except ValueError:
                        t.error(f"SYSTEM: Invalid INJSIM parameters: {parts[1:]}")
                        sys.exit("SYSTEM: Aborting IFPmin...")
                    injsim_params = {'A': A, 'Fb': Fb}
                else:
                    sim[key] = (len(parts) == 1) or (parts[1].upper() in ['TRUE','T','YES','1'])
            i += 1
            
        self.config['simtype'] = sim
        self.config['simtype_order'] = order
        if injsim_params:
            self.config['injsim_params'] = injsim_params

        # return index after SIMTYPE /
        for j in range(i, len(lines)):
            if lines[j].upper().startswith('SIMTYPE') and '/' in lines[j]:
                return j+1
            
        return len(lines)

    def _parse_resprop(self, lines, start):
        v   = lines[start].split()
        idx = start + 1
        self.config['resprop'] = {
            'pressure_bar':           float(v[0]),
            'frac_pressure_bar':      float(v[1]),
            'temperature_C':          float(v[2]),
            'rock_compressibility_1_per_bar': float(v[3]),
            'water_compressibility_1_per_bar':float(v[4]),
            'flow_regime':            v[5]
        }

        if lines[idx].upper().startswith('PETRO'):
            idx += 1
            p  = list(map(float, lines[idx].split())); idx += 1
            k  = list(map(float, lines[idx].split())); idx += 1
            net= list(map(float, lines[idx].split())); idx += 1
            self.config['resprop']['petro'] = {
                'porosity_min':   p[0], 'porosity_avg':   p[1], 'porosity_max':   p[2],
                'permeability_min_mD': k[0], 'permeability_avg_mD': k[1], 'permeability_max_mD': k[2],
                'netpay_min_m':   net[0], 'netpay_avg_m':   net[1], 'netpay_max_m':   net[2]
            }
            while idx < len(lines) and not (lines[idx].upper().startswith('PETRO') and '/' in lines[idx]):
                idx += 1
            idx += 1

        return next((j+1 for j in range(idx, len(lines))
                     if lines[j].upper().startswith('RESPROP') and '/' in lines[j]),
                    len(lines))

    def _parse_wellspec(self, lines, start):
        v = lines[start].split()
        self.config['wellspec'] = {
            'water_depth_m':        float(v[0]),
            'depth_below_sl_m':     float(v[1]),
            'wellbore_radius_m':    float(v[2]),
            'injection_pressure_bar':float(v[3])
        }

        return next((j+1 for j in range(start, len(lines))
                     if lines[j].upper().startswith('WELLSPEC') and '/' in lines[j]),
                    len(lines))

    def _read_mineralogy(self):
        path = os.path.join(self.input_dir, 'Mineralogy.min')

        with open(path, encoding='utf-8-sig') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        try:
            s = lines.index('COMPS')
            e = lines.index('COMPS /')

        except ValueError:
            t.error('SYSTEM: Mineralogy.min COMPS missing'); sys.exit("SYSTEM: Aborting IFPmin...")

        head = lines[s+1].split()
        data = lines[s+2:e]

        if len(data) < 25:
            t.warning('SYSTEM: Mineralogy <25 observations; results might be biased!')

        obs = []
        for r in data:
            vals = list(map(float, r.split()))
            if len(vals) != len(head):
                t.error('SYSTEM: Mineralogy row mismatch; please control your input!')
                sys.exit("SYSTEM: Aborting IFPmin...")

            obs.append(dict(zip(head, vals)))

        self.config['mineralogy'] = {
            'compounds':      head,
            'observations':   obs,
            'num_observations': len(obs)
        }
        t.info(f"SYSTEM: Parsed {len(obs)} mineralogy records")

    def _build_template(self, include_dolo: bool):
        dolo = ("DOLOMITE\nDOLOMITE-DIS\nDOLOMITE-ORD\nDOLOMITE-sed\n") if include_dolo else ''
        return f"""
TEST
COMPUTE SPC
COMPUTE EQU
COMPUTE Q
END TEST

CONDITIONS
TITLE  \"Simulation {{sim_num}}\"  
OUTPUT out/sim{{sim_num}}  
END CONDITIONS

INCLUDE dtb\\elements.dtb
INCLUDE dtb\\hkf_aqu.dtb
INCLUDE dtb\\hkf_gas.dtb
INCLUDE dtb\\hkf_min.dtb

SOLVENT
MODEL DH2EQ3
END SOLVENT

SYSTEM
TdgC  {{temperature}}
Pbar  {{pressure}}
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
MOLE AMORPH-SILICA    {{AMPS}}
!MOLE RUTILE           {{RUT}}
MOLE CORUNDUM         {{COR}}
MOLE HEMATITE         {{HEM}}
MOLE FERROUS-OXIDE    {{FER}}
MOLE MANGANOSITE      {{MAN}}
MOLE PERICLASE        {{PER}}
MOLE LIME             {{LIME}}
MOLE SODIUM-OXIDE     {{SOD}}
MOLE POTASSIUM-OXIDE  {{POT}}
!MOLE P2O3             {{P2O3}}
END SYSTEM.ROCK

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
{dolo}
END SPECIES.EXCLUDE
"""
    
    def _generate_simulation_files(self,minerology_df, res_pres, res_temp,
                                   output_subdir='simulations', num_simulation=1000, include_dolo=True):
        sim_dir=os.path.join(self.output_dir,output_subdir); os.makedirs(sim_dir,exist_ok=True)
        tpl=self._build_template(include_dolo)
        rng=np.random.default_rng(self.seed)
        self.random_mineralogy_df=pd.DataFrame()
        compounds=minerology_df.columns.tolist()
        rows=[]

        for num in range(1,num_simulation+1):
            weights=rng.random(len(compounds));props=weights/weights.sum()
            vals=dict(zip(compounds,props))
            rows.append(vals)
            content=tpl.format(sim_num=num,temperature=res_temp,pressure=res_pres,**vals)
            
            with open(os.path.join(sim_dir,f"sim{num:04d}.inn"),'w') as f: f.write(content)
            t.info(f"CRTSIM: Generated → sim{num:04d}.inn")

        t.info(f"Generated {num_simulation} synthetic sim files in {sim_dir}")
        self.random_mineralogy_df=pd.DataFrame(rows)
    
    def _validate_crtsim(self):
        path = os.path.join(self.input_dir, 'Mineralogy.min')
        if not os.path.isfile(path):
            t.error(f'SYSTEM: Mineralogy.min not found at {path}!')
            sys.exit("SYSTEM: Aborting IFPmin...")

        # read lines and strip comments/blank
        lines = [ln.strip() for ln in open(path, encoding='utf-8-sig') if ln.strip()]
        try:
            start = lines.index('COMPS') + 1
            end   = lines.index('COMPS /')

        except ValueError:
            t.error('CRTSIM: COMPS section is missing in Mineralogy.min!')
            sys.exit("SYSTEM: Aborting IFPmin...")

        n_obs = len(lines[start:end])
        if n_obs < 3:
            t.error(f'CRTSIM: only {n_obs} mineralogy observations (<3)')
            sys.exit("Aborting IFPmin...")

        # check resprop entries
        rp = self.config.get('resprop', {})
        for key in ('pressure_bar','temperature_C'):
            if key not in rp:
                t.error(f'CRTSIM: RESPROP missing "{key}"')
                sys.exit("SYSTEM: Aborting IFPmin...")

        return n_obs, rp

    def _validate_tdynsim(self):
        order = self.config['simtype_order']

        if 'CRTSIM' not in order or order.index('CRTSIM') > order.index('TDYNSIM'):
            t.error('SYSTEM: CRTSIM must precede TDYNSIM!'); sys.exit("SYSTEM: Aborting IFPmin...")

        sim_dir = os.path.join(self.output_dir, 'simulations')

        if not os.path.isdir(sim_dir) or not os.listdir(sim_dir):
            t.error('SYSTEM: No simulation files for TDYNSIM!'); sys.exit("SYSTEM: Aborting IFPmin...")

        if not os.path.isfile(self.arxim_exe):
            t.error('SYSTEM: ArXim executable not found!'); sys.exit("SYSTEM: Aborting IFPmin...")

    def _run_thermodynamic_simulations(self):
        inp = os.path.join(self.output_dir, 'simulations')
        # ensure ArXim's own "out" lives alongside the .exe
        os.makedirs(os.path.join(self.arxim_dir, 'out'), exist_ok=True)

        for fn in sorted(f for f in os.listdir(inp) if f.endswith('.inn')):
            full_inn = os.path.join(inp, fn)
            out_line = next((ln for ln in open(full_inn, encoding='utf-8-sig')
                            if 'OUTPUT' in ln), None)
            
            if not out_line:
                t.warning(f"TDYNSIM: Skipping {fn}: no OUTPUT directive!")
                continue

            t.info(f"TDYNSIM: ArXim {fn} → (arxim_dir/out)")

            try:
                subprocess.run(
                    [self.arxim_exe, full_inn],
                    cwd=self.arxim_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

            except subprocess.CalledProcessError as e:
                t.error(f"ArXim failed on {fn} (exit {e.returncode}). Skipping. stderr:\n{e.stderr.decode()}")
                # optionally continue to next file
                continue

            t.info('TDYNSIM: Completed ArXim runs!')

    def _extract_equilibrium_species(self):
        res_dir  = os.path.join(self.arxim_dir, 'out')
        proc_dir = os.path.join(self.output_dir, 'Processed')
        os.makedirs(proc_dir, exist_ok=True)
        combined = os.path.join(proc_dir, 'Combined_Equilibrium_Species.txt')

        files = sorted(
            [f for f in os.listdir(res_dir)
             if re.fullmatch(r'sim\d+_equil\.res', f)],
            key=lambda x: int(re.search(r'\d+', x).group())
        )

        if not files:
            t.warning('TDYNSIM: No .res files to process in arxim_dir/out!'); return

        with open(combined, 'w', encoding='utf-8-sig') as outf:
            for f in files:
                outf.write(f"File: {f}\n")
                lines = open(os.path.join(res_dir, f),
                             encoding='utf-8-sig').readlines()

                in_block = False
                found    = False
                for ln in lines:
                    if 'Equilibrium Species' in ln:
                        in_block = True
                        found    = True
                    if in_block:
                        outf.write(ln)

                    if in_block and ln.strip().startswith('_'):
                        break

                if not found:
                    t.warning(f"TDYNSIM: No equilibrium block in {f}!")

                outf.write('\n' + '_'*60 + '\n')

        t.info(f"TDYNSIM: Aggregated → {combined}")

    def _analyze_results(self):
        proc_dir = os.path.join(self.output_dir, 'Processed')
        combined = os.path.join(proc_dir, 'Combined_Equilibrium_Species.txt')
        excel    = os.path.join(proc_dir, 'Results.xlsx')

        if not os.path.isfile(combined):
            t.error('TDYNSIM: Combined txt missing!'); return

        rows    = []
        sim_no  = None
        minerals= {}
        for ln in open(combined, encoding='utf-8-sig'):
            if ln.startswith('File:'):
                sim_no = int(re.search(r'\d+', ln).group())
                minerals = {}
            elif 'MOLE=' in ln:
                parts = ln.split()
                minerals[parts[0]] = float(parts[-1].split('=')[-1])
            elif ln.strip().startswith('_') and sim_no is not None:
                rows.append({'Simulation': sim_no, **minerals})
                sim_no = None

        if rows:
            df = pd.DataFrame(rows).fillna(0)
            df.to_excel(excel, index=False)
            t.info(f"TDYNSIM: Analysis processed and saved at {excel}!")

        else:
            t.warning('TDYNSIM: No data in combined file')

    def _count_successful_simulations(self):
        out_dir = os.path.join(self.arxim_dir, 'out')
        count   = sum(1 for f in os.listdir(out_dir)
                      if f.endswith('_equil.res'))
        sim_dir = os.path.join(self.output_dir, 'simulations')
        total   = len([f for f in os.listdir(sim_dir) if f.endswith('.inn')])
        pct     = count/total*100 if total else 0

        t.info(f"TDYNSIM: Successful simulations: {count}/{total} ({pct:.1f}%)")

    def _validate_stgsim(self):
        order=self.config['simtype_order']
        if 'TDYNSIM' not in order or order.index('TDYNSIM')>order.index('STGSIM'):
            # if TDYNSIM not before STGSIM, require existing results
            proc=os.path.join(self.output_dir,'Processed','Results.xlsx')
            if not os.path.isfile(proc): t.error('TDYNSIM: Results.xlsx missing'); sys.exit("SYSTEM: Aborting IFPmin...")
        
    def _validate_injsim(self):
        """
        Ensure INJSIM is correctly configured: CRTSIM and TDYNSIM must precede it,
        and A/Fb parameters must be present.
        """
        if 'INJSIM' not in self.config.get('simtype', {}):
            t.error('INJSIM flag missing'); sys.exit("SYSTEM: Aborting IFPmin...")

        # Check parameters
        params = self.config.get('injsim_params')
        if not params or not all(k in params for k in ('A','Fb')):
            t.error('INJSIM parameters A and Fb missing!'); sys.exit("SYSTEM: Aborting IFPmin...")

        # Validate RESPROP entries
        rp = self.config.get('resprop', {})
        required_rp = ['pressure_bar','frac_pressure_bar','temperature_C',
                       'rock_compressibility_1_per_bar','water_compressibility_1_per_bar','flow_regime']
        missing_rp = [k for k in required_rp if k not in rp]

        if missing_rp:
            t.error(f"RESPROP missing parameters: {missing_rp}"); sys.exit('SYSTEM: Aborting IFPmin...')

        if 'petro' not in rp:
            t.error('RESPROP: PETRO block missing!'); sys.exit('SYSTEM: Aborting IFPmin...')

        petro = rp['petro']
        required_petro = ['porosity_min','porosity_avg','porosity_max',
                          'permeability_min_mD','permeability_avg_mD','permeability_max_mD',
                          'netpay_min_m','netpay_avg_m','netpay_max_m']
        
        missing_p = [k for k in required_petro if k not in petro]

        if missing_p:
            t.error(f"PETRO missing fields: {missing_p}"); sys.exit('SYSTEM: Aborting IFPmin...')

        # Validate WELLSPEC entries
        ws = self.config.get('wellspec', {})
        required_ws = ['water_depth_m','depth_below_sl_m','wellbore_radius_m','injection_pressure_bar']
        missing_ws = [k for k in required_ws if k not in ws]

        if missing_ws:
            t.error(f"WELLSPEC missing parameters: {missing_ws}"); sys.exit('SYSTEM: Aborting IFPmin...')

        return params

    def _validate(self):
        """
        Final overall checks: ensure each requested simulation phase produced expected files.
        """
        sim = self.config.get('simtype', {})
        # CRTSIM: .inn files must exist
        if sim.get('CRTSIM'):
            sim_dir = os.path.join(self.output_dir, 'simulations')
            if not os.path.isdir(sim_dir) or not any(f.endswith('.inn') for f in os.listdir(sim_dir)):
                t.error('CRTSIM: No .inn files found; CRTSIM did not run correctly!')
                sys.exit('SYSTEM: Aborting CRTSIM outputs missing')

        # TDYNSIM: .res files must exist
        if sim.get('TDYNSIM'):
            res_dir = os.path.join(self.arxim_dir, 'out')
            if not os.path.isdir(res_dir) or not any(f.endswith('.res') for f in os.listdir(res_dir)):
                t.error('TDYNSIM: No .res files found; TDYNSIM did not run correctly!')
                sys.exit('SYSTEM: Aborting IFPmin...')

        # STGSIM: storage results must exist
        if sim.get('STGSIM'):
            store_file = os.path.join(self.output_dir, 'Processed', 'Storage_Results.xlsx')
            if not os.path.isfile(store_file):
                t.error('STGSIM: Storage_Results.xlsx missing; STGSIM did not run correctly!')
                sys.exit('SYSTEM: Aborting IFPmin...')

        # INJSIM: injectivity results must exist
        if sim.get('INJSIM'):
            inj_file = os.path.join(self.output_dir, 'Processed', 'Injectivity_Results.xlsx')
            if not os.path.isfile(inj_file):
                t.error('INJSIM: Injectivity_Results.xlsx missing; INJSIM did not run correctly!')
                sys.exit('SYSTEM: Aborting IFPmin...')

        t.info('SYSTEM: All requested simulation phases produced expected outputs!')

    def _process_simulation_data(self):
        proc = os.path.join(self.output_dir,'Processed'); results=pd.read_excel(os.path.join(proc,'Results.xlsx'))
        miner = self.random_mineralogy_df.copy()
        miner['Simulation Number'] = miner.index+1

        ivr = (miner.get('AMPS', 0) * 27.30 + miner.get('RUT', 0) * 26.10 + miner.get('COR', 0) * 25.58 +miner.get('HEM', 0) * 30.30 +
             miner.get('FER', 0) * 12.00 + miner.get('MAN', 0) * 13.22 + miner.get('PER', 0) * 11.25 + miner.get('LIME') * 16.76 + 
             miner.get('SOD', 0) * 27.75 + miner.get('POT', 0) * 41.04 + miner.get('P2O3',0) * 51.50) / 1e6
        
        miner['Initial Volume Rock, m3'] = ivr

        df = results.merge(miner[['Simulation Number','Initial Volume Rock, m3']], on='Simulation Number')
      
        n = len(df); rp = self.config['resprop']['petro']
        df['Porosity'] = np.random.triangular(rp['porosity_min'], rp['porosity_avg'], rp['porosity_max'], n)

        for name, l, m, r in [
            ('Ea', 0.5, 0.65, 0.8),
            ('Ev', 0.6, 0.75, 0.9),
            ('Ed', 0.5, 0.65, 0.8),
            ('Eg', 0.25, 0.625, 1.0),
            ('Em', 0.4, 0.6, 0.8)
        ]:
            df[name] = np.random.triangular(l, m, r, n)

        df['E'] = (df['Ea'] * df['Ev'] * df['Ed'] * df['Eg']) / (1 - df['Em'])
        # Clamp efficiency factor to physical limit
        df.loc[df['E'] >= 0.9, 'E'] = 0.9

        # CO2 mass
        inc=self.config['simtype'].get('TDYNSIM',False)
        df['kg CO2'] = (44.01 * df.get('CALCITE', 0) + (44.01 * df.get('DOLOMITE-ORD', 0) * 2 if inc else 0)
                      + 44.01 * df.get('RHODOCHROSITE', 0) + 44.01 * df.get('MAGNESITE', 0)) / 1000
        
        # final volume: use resulting minerals
        factors = {
            'ALUNITE': 45.00,'CALCITE': 36.93,'GIBBSITE': 31.96,'MAGNESITE': 28.02,
            'MUSCOVITE': 140.81,'RHODOCHROSITE': 31.10,'HEMATITE': 30.30,'MICROCLINE': 108.70,
            'QUARTZ-A': 22.69,'ANTIGORITE': 24.80, 'AMORPH-SILICA': 27.30, 'DOLOMITE-ORD': 36.90, 
            'PARAGONITE':132.53, 'FERROUS-OXIDE': 12.60, 'ALBITE-HIGH': 100.07, 'ALBITE-LOW': 100.07, 
            'LAUMONITE': 201.00, 'TALC': 138.00, 'ANTIGORITE': 1754.80, 'CLINOCHLORE-14A': 213.00
        }

        # Sum volumes from output species columns
        df['Final Volume Rock, m3'] = sum(df.get(k, 0) * v for k, v in factors.items()) / 1e6

        # Storage capacity estimation
        df['Storage, kg CO2/m3'] = (df['kg CO2'] / df['Final Volume Rock, m3']) * df['Porosity'] * df['E']

        df['Above_Line'] = df['Initial Volume Rock, m3'] > df['Final Volume Rock, m3']
        df['Above_Line_Label'] = df['Above_Line'].map({True:'Ratio > 1',False:'Ratio < 1'})

        outp = os.path.join(proc,'Storage_Results.xlsx'); df.to_excel(outp, index=False)

        t.info(f"STGSIM: Storage Capacity Simulation is done; data saved to {outp}!")

        return df
    
    def _plot_ratio_comparison(self, df):
        """
        Create a dissolution vs precipitation comparison plot.

        Args:
            df (DataFrame): DataFrame to plot initial and final volumes.
        """

        self.proc = os.path.join(self.output_dir, 'Analysis Plots')
        os.makedirs(self.proc, exist_ok=True)

        palette = {'Ratio > 1': 'red', 'Ratio < 1': 'green'}

        # Determine plotting range
        min_val = min(
            df['Final Volume Rock, m3'].min(),
            df['Initial Volume Rock, m3'].min()
        )
        max_val = max(
            df['Final Volume Rock, m3'].max(),
            df['Initial Volume Rock, m3'].max()
        )
        buffer = (max_val - min_val) * 0.05
        min_val -= buffer
        max_val += buffer

        plt.figure(figsize=(16, 9))
        sns.scatterplot(
            data=df,
            x='Final Volume Rock, m3',
            y='Initial Volume Rock, m3',
            hue='Above_Line_Label',
            palette=palette
        )

        # 45-degree reference line
        x_vals = np.linspace(min_val, max_val, 100)
        plt.plot(x_vals, x_vals, color='black', linestyle='--', label='Ratio = 1')

        # Labeling, grid, style
        plt.xlabel('Final Precipitated Volume [m³]')
        plt.ylabel('Initial Mineral Volume [m³]')

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.annotate(
            r'$\text{Ratio} = \frac{\text{Initial}}{\text{Final}}$',
            xy=(min_val + 0.2 * (max_val - min_val), max_val - 0.1 * (max_val - min_val)),
            fontsize=20,
            bbox=dict(boxstyle="round", fc="white", ec="black")
        )

        plt.legend(loc='upper left')
        plt.grid(alpha=0.5)
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20

        plt.tight_layout()
        save_path = os.path.join(self.proc, 'Dissolution_vs_Precipitation.pdf')
        plt.savefig(save_path, format='pdf')

        t.info(f"SGTSIM: Dissolution - Precipitation comparison analysis saved to {save_path}!")
    
    def _SimulateStorageCapacity(self, df):
        """
        Simulates random storage scenarios based on Monte-Carlo simulation with the usage of
        Latin Hypercube Sampling.

        Args:
            df (DataFrame): Therymodyanmic simulation results from TDYNSIM.
        """

        # --- 1) SAMPLE INPUTS ---
        # prepare sorted empirical arrays
        emp = {
            'co2': np.sort(df['kg CO2'].values),
            'vol': np.sort(df['Initial Volume Rock, m3'].values),
            'poro': np.sort(df['Porosity'].values),
            'E': np.sort(df['E'].values),
        }
        sampler = qmc.LatinHypercube(d=4, seed=self.seed)
        u = sampler.random(1000)  # uniform [0,1)^4
        # map uniforms through empirical CDFs
        random_co2  = np.interp(u[:, 0], np.linspace(0,1,len(emp['co2'])),  emp['co2'])
        random_vol  = np.interp(u[:, 1], np.linspace(0,1,len(emp['vol'])),  emp['vol'])
        random_poro = np.interp(u[:, 2], np.linspace(0,1,len(emp['poro'])), emp['poro'])
        random_E    = np.interp(u[:, 3], np.linspace(0,1,len(emp['E'])),    emp['E'])

        # --- 2) COMPUTE STORAGE & GRV ---
        # Storage Capacity in kg CO2/m3 rock, GRV in m3 rock per target tonne
        storage = (random_co2 / random_vol) * (1 - random_poro) * random_E
        grv     = (self.config['target'].get('storage_objective_t_per_year') * 1000) / storage
        t.info(f"STGSIM: Target → {round(self.config['target'].get('storage_objective_t_per_year') / 1e6)} Mt")

        # --- 3) EXTRACT QUANTILES & ASSOCIATED SCENARIOS ---
        p10 = np.percentile(storage, 10)
        p50 = np.percentile(storage, 50)
        p90 = np.percentile(storage, 90)

        # Find the nearest sample for each percentile
        def pick(q_val):
            idx = np.argmin(np.abs(storage - q_val))
            return {
                'storage': storage[idx],
                'grv':      grv[idx],
                'poro':     random_poro[idx],
                'E':        random_E[idx],
            }

        scen10 = pick(p10)
        scen50 = pick(p50)
        scen90 = pick(p90)

        # --- 4) PLOT ECDF ---
        os.makedirs(self.proc, exist_ok=True)
        plt.figure(figsize=(16, 9))
        sns.set_palette('husl', 8)

        ax = sns.ecdfplot(storage, complementary=True, linewidth=2)
        ax.set_xlabel(r'Storage Capacity [kg $\mathrm{CO_2}/\mathrm{m}^3$]')
        ax.set_ylabel('Probability')
        ax.grid(alpha=0.5)
        plt.rcParams.update({'font.size': 18, 'font.family': 'Arial'})
        plt.xlim(storage.min(), storage.max())

        def annotate_point(scn, q_label, prob, xytext):
            txt = (
                f"{q_label}\n"
                f"$C_{{CO_2}}$ = {round(scn['storage']):.2f} kg/m³\n"
                f"ϕ = {round(scn['poro'] * 100):.2f}%\n"
                f"E = {round(scn['E'] * 100):.2f}%\n"
                f"GRV = {round(scn['grv']):.2e} m³"
            )
            
            ax.annotate(txt,
                        xy=(scn['storage'], prob),
                        xytext=xytext,
                        bbox=dict(boxstyle="round", fc="0.8"),
                        arrowprops=dict(arrowstyle="->"))

            # lines
            ax.vlines(x=scn['storage'], ymin=0, ymax=prob,
                        colors=('red' if q_label=='P90' else
                                'blue' if q_label=='P50' else
                                'green'),
                        linestyles='--')
            ax.hlines(y=prob, xmin=0, xmax=scn['storage'],
                        colors=('red' if q_label=='P90' else
                                'blue' if q_label=='P50' else
                                'green'),
                        linestyles='--')

        annotate_point(scen10, 'P90', 0.9, (scen10['storage']*1.05, 0.8))
        annotate_point(scen90, 'P10', 0.1, (scen90['storage']*1.05, 0.2))
        annotate_point(scen50, 'P50', 0.5, (scen50['storage']*1.05, 0.5))

        plt.xlim(0)

        fname = 'Monte_Carlo(LHS)_Storage_Capacity.pdf'
        save_path = os.path.join(self.proc, fname)
        plt.rcParams['font.size'] = 20
        plt.rcParams['font.family'] = 'Arial'
        
        plt.tight_layout()
        plt.savefig(save_path, format='pdf')

        t.info(f"STGSIM: Storage Capacity Estimation analysis saved to {save_path}!")

    def _SimulateInjectivity(self):
        from InjectionFP import IFlowP
        # uniform LHS in 3 dims
        sampler = qmc.LatinHypercube(d=3, seed=self.seed)
        u = sampler.random(10000)
        # map to triangular via scipy.stats.triang.ppf
        por_dist = triang(c=(self.config['resprop']['petro'].get('porosity_avg')-self.config['resprop']['petro'].get('porosity_min'))/(self.config['resprop']['petro'].get('porosity_max') - self.config['resprop']['petro'].get('porosity_min')),
                            loc=self.config['resprop']['petro'].get('porosity_min'), scale=(self.config['resprop']['petro'].get('porosity_max') - self.config['resprop']['petro'].get('porosity_min')))
        k_dist   = triang(c=(self.config['resprop']['petro'].get('permeability_avg_mD')  - self.config['resprop']['petro'].get('permeability_min_mD')) / (self.config['resprop']['petro'].get('permeability_max_mD')  - self.config['resprop']['petro'].get('permeability_min_mD')),
                            loc=self.config['resprop']['petro'].get('permeability_min_mD'),   scale=(self.config['resprop']['petro'].get('permeability_max_mD')  - self.config['resprop']['petro'].get('permeability_min_mD')))
        h_dist   = triang(c=(self.config['resprop']['petro'].get('netpay_avg_m')  - self.config['resprop']['petro'].get('netpay_min_m')) / (self.config['resprop']['petro'].get('netpay_max_m')  - self.config['resprop']['petro'].get('netpay_min_m')),
                            loc=self.config['resprop']['petro'].get('netpay_min_m'), scale=(self.config['resprop']['petro'].get('netpay_max_m')  - self.config['resprop']['petro'].get('netpay_min_m')))

        por_samples = por_dist.ppf(u[:,0])
        k_samples   = k_dist.ppf(  u[:,1])
        h_samples   = h_dist.ppf(  u[:,2])

        # avoid zero-perm
        k_samples = np.clip(k_samples, 1e-3, None)

        # 2) RUN ALL SIMULATIONS
        II_list     = np.empty(10000)
        wells_list  = np.empty(10000, dtype=int)
        volume_list = np.empty(10000)
        dp_list     = np.empty(10000)
        effective_skin_list = np.empty(10000)
        sufficient_list = np.empty(10000)

        for i in range(10000):
            flow = IFlowP(self)
            
            ii, volme, wells, dp, qg, effective_skin, sufficient, _, _ = flow.co2_injectivity(
                poro=por_samples[i],
                k = k_samples[i],
                h = h_samples[i],
                rw= self.config['wellspec'].get('wellbore_radius_m')
            )

            II_list[i]     = ii
            wells_list[i]  = wells
            volume_list[i] = volme
            dp_list[i]     = dp
            effective_skin_list[i] = effective_skin
            sufficient_list[i] = sufficient

        df_injectivity = pd.DataFrame({
            'II [tonnes CO2.cp/bar^2.year]': II_list,
            'Number of Wells': wells_list,
            'Storable CO2 Volume [tonnes]': volume_list,
            'DP [bar^2.cp]': dp_list,
            'Effective Skin [-]': effective_skin_list,
            'Is Project Sufficient?': sufficient_list
        })

        df_injectivity['Is Project Sufficient?'] = df_injectivity['Is Project Sufficient?'] <= 1.0

        idx50 = np.abs(II_list - np.percentile(II_list,50)).argmin()

        single_poros  = por_samples[idx50]
        single_ks     = k_samples[idx50]
        single_hs     = h_samples[idx50]

        p_res, time_span = self._generate_single_history(
            single_poros, single_ks, single_hs,
            self.config['wellspec']['wellbore_radius_m']
        )

        proc_excel = os.path.join(self.output_dir,'Processed')
        outp = os.path.join(proc_excel,'Injectivity_Results.xlsx'); df_injectivity.to_excel(outp, index=False)

        t.info(f"INJSIM: Injectivity Simulation is done; data saved to {outp}!")

        # 3) COMPUTE P10, P50, P90 + nearest-scenario lookups
        p10 = np.percentile(II_list, 10)
        p50 = np.percentile(II_list, 50)
        p90 = np.percentile(II_list, 90)

        def pick(q):
            idx = np.abs(II_list - q).argmin()
            return {
                'II':       II_list[idx],
                'wells':    wells_list[idx],
                'volume':   volume_list[idx],
                'Δp':       dp_list[idx],
                'porosity': por_samples[idx],
                'k':        k_samples[idx],
                'h':        h_samples[idx]
            }

        scen10 = pick(p10)
        scen50 = pick(p50)
        scen90 = pick(p90)

        # 4) PLOT complementary-CDF with annotations
        fname = 'Monte_Carlo(LHS)_Injectivity.pdf'
        save_path = os.path.join(self.proc, fname)

        plt.figure(figsize=(16, 9))
        sns.set_palette('husl', 8)

        ax = sns.ecdfplot(II_list, complementary=True, linewidth=2)
        ax.set_xlabel(r'Injectivity Index [tonnes $\mathrm{CO_2}\!\cdot\!cp/\mathrm{bar}^2\!\cdot\!year$]')
        ax.set_ylabel('Probability')
        plt.grid(alpha=0.5)
        plt.xlim(II_list.min(), II_list.max())

        def ann(scn, label, prob, xytext):
            txt = (
                f"{label}\n"
                f"II = {scn['II']:.2f} tonnes CO2·cp/bar²·year\n"
                f"Wells = {scn['wells']}\n"
                f"k = {round(scn['k'])} mD\n"
                f"h = {round(scn['h'])} m\n"
                f"V = {scn['volume']:.2e} tonnes CO2"
            )
            ax.annotate(
                txt,
                xy=(scn['II'], prob),
                xytext=xytext,
                bbox=dict(boxstyle="round", fc="0.8"),
                arrowprops=dict(arrowstyle="->")
            )
            color = {'P90':'red', 'P50':'blue', 'P10':'green'}[label]
            ax.vlines(scn['II'], 0, prob, colors=color, linestyles='--')
            ax.hlines(prob, 0, scn['II'], colors=color, linestyles='--')


        ann(scen90, 'P10', 0.1, (scen90['II']*1.05, 0.2))
        ann(scen50, 'P50', 0.5, (scen50['II']*1.05, 0.5))
        ann(scen10, 'P90', 0.9, (scen10['II']*1.05, 0.8))

        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 18

        plt.tight_layout()
        plt.savefig(save_path, format='pdf')

        t.info(f"INJSIM: Injectivity Estimation analysis saved to {save_path}!")

        return time_span, p_res
    
    def _generate_single_history(self, poro, k, h, rw):
        from InjectionFP import IFlowP
        flow = IFlowP(self)

        II, V, wells, dp, Qg, skin, suff, p_res, time_span = flow.co2_injectivity(poro, k, h, rw)

        return p_res, time_span

    def _plot_pressure(self, time, pressure):
        # enforce initial conditions
        time[0] = 0
        pressure[0] = self.config['resprop']['pressure_bar']

        # put data into a DataFrame for Seaborn
        df = pd.DataFrame({"Months": time, "Pressure (bar)": pressure})

        # apply a seaborn theme
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # create the figure
        plt.figure(figsize=(16, 9))
        sns.lineplot(data=df, x="Months", y="Pressure (bar)",
                    color="orange", linewidth=2, label="Pres")

        # add the horizontal frac‐pressure line
        plt.axhline(self.config['resprop']['frac_pressure_bar'], color="red", linestyle="--", linewidth=2, label="Pfrac")

        # axis labels & limits
        plt.xlabel("Time [year]")
        plt.ylabel("Pressure [bar]")

        y_max = max(max(pressure), self.config['resprop']['frac_pressure_bar'])

        plt.ylim(self.config['resprop']['pressure_bar'], math.ceil(1.1 * y_max))
        plt.xlim(0, max(time))

        plt.legend(loc="upper left", frameon=True)

        plt.rcParams['font.size'] = 18
        plt.rcParams['font.family'] = 'Arial'

        plt.tight_layout()

        fname = 'Monte_Carlo(LHS)_P50_PBU.pdf'
        save_path = os.path.join(self.proc, fname)

        plt.savefig(save_path, format='pdf')
        
        t.info(f"INJSIM: P50 Injectivity Estimation Analysis pressure build up saved to {save_path}!")

    def get(self, section: str):
        return self.config.get(section)

if __name__ == '__main__':
    cfg = CO2InjectionConfig()


