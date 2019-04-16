"""Abstract model of enteric disease transmission and behavioral adaptation among households under variable
environmental conditions.

When setting up parameter sweeps, it is possible to distinguish between 'base' and 'experimental' parameters for the
purpose of seeding the random-number generator. Combinations of base parameter values will be replicated using different
random seeds from a list that is repeated across combinations of experimental parameter values.

Lists of values should be specified for the following parameters within main():
    rec_rate (float): Recovery rate (recovery events/infected individual).
    inf_rate (float): Infection rate (infections/pathogen).
    contam (float): Environmental contamination level (pathogens).
    hh_n (int): Size of households (individuals).
    net_n (int): Size of social network (households).
    net_k (int): Network degree (edges/node)
    net_p (float): Edge rewiring probability (parameter required to generate random small-world networks according to
        the Watts-Strogatz model).
    optimal_init (float): Proportion of households initialized with the optimal strategy.
    eff_diff (float): Difference in strategy efficacy (β parameter for distribution governing the efficacy of the
        weather-sensitive strategy).
    duration (int): Duration of environmental cycles (days).
    num_cycles (int): Number of environmental cycles to run.
    scenario (str): Environmental change scenario ('baseline': weather-sensitive strategy always optimal; 'punctuated':
        weather-sensitive strategy changes from optimal during the first cycle to suboptimal thereafter.

"""

import itertools as itrt
from multiprocessing import Pool
import networkx as nx
import numpy as np
import os
import pandas as pd
import shutil
import time as sys_time


def main():
    """Defines parameter values for experiments."""
    output_dir = 'model_final_actual/'  # output directory path
    num_processors = 12  # number of available processors to run experiments
    num_replicates = 50  # number of replicates for each uniquely parameterized model run

    base_params = {'rec_rate': [0.1],
                   'inf_rate': [0.000002],
                   'contam': [2000.0],
                   'hh_n': [4],
                   'net_n': [80],
                   'net_p': [0.3],
                   'optimal_init': [0.025, 0.05, 0.075],
                   'eff_diff': [0.05, 0.5, 0.95],
                   'duration': [180, 365],
                   'num_cycles': [3],
                   'scenario': ['baseline', 'punctuated']}  # base parameters
    exp_params = {'net_k': [6, 12]}  # experimental parameters

    run_experiments(output_dir, num_processors, num_replicates, base_params, exp_params)


def run_experiments(output_dir, num_processors, num_replicates, base_params, exp_params):
    """Sets up and runs experiments, saves dataframes of parameter values and output.

    Args:
        output_dir (str): Output directory path.
        num_processors (int): Number of available processors to run experiments.
        num_replicates (int): Number of replicates for each uniquely parameterized model run.
        base_params (dict): Dictionary of base parameter names and values.
        exp_params (dict): Dictionary of experimental parameter names and values.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_params['replicate'] = [int(_) for _ in np.linspace(0, num_replicates - 1, num_replicates)]
    base_combinations = list(itrt.product(*base_params.values()))  # Cartesian product of base_params values
    base_total = len(base_combinations)
    exp_combinations = itrt.product(*exp_params.values())  # Cartesian product of exp_params values
    rng_seeds = [int(_) for _ in np.linspace(500000, 500000 + (base_total - 1) * 1000, base_total)]
    param_names = list(base_params.keys()) + list(exp_params.keys()) + ['run_id', 'rng_seed']

    t1 = sys_time.time()
    exp_id = 0  # experiment ID (combined with RNG seeds to create unique model run IDs)
    for exp_combination in exp_combinations:
        param_dicts = gen_param_dicts(param_names, base_combinations, exp_combination, exp_id, rng_seeds)
        pd.DataFrame(param_dicts).to_csv(output_dir + 'params_exp' + str(exp_id), index=False)
        pool = Pool(processes=num_processors, maxtasksperchild=1000)
        output_dicts = pool.map(gen_run, param_dicts, chunksize=15)  # output from individual model runs
        pool.close()
        pool.join()
        output_dicts = [pd.DataFrame(_) for _ in output_dicts]
        pd.concat(output_dicts, ignore_index=True).to_csv(output_dir + 'output_exp' + str(exp_id), index=False)
        exp_id += 1
    shutil.copy(__file__, output_dir)
    print(sys_time.time() - t1)


def gen_param_dicts(param_names, base_combinations, exp_combination, exp_id, rng_seeds):
    """Generates list of parameter dictionaries required to run model.

    Args:
        param_names (list): Names of parameters.
        base_combinations (list): Combinations of base parameter values.
        exp_combination (list): Experiment-specific parameter values.
        exp_id (int): Experiment ID.
        rng_seeds (list): RNG seeds for individual model runs making up the experiment.

    Returns:
        param_dicts (list): Dictionaries of parameter values for individual model runs.

    """
    param_dicts = []
    for i, base_list in enumerate(base_combinations):
        vals = base_list + exp_combination + ('_'.join([str(exp_id), str(rng_seeds[i])]), rng_seeds[i])
        param_dicts.append(dict(zip(param_names, vals)))
    return param_dicts


def gen_run(param_dict):
    """Instantiates model run.

    Args:
        param_dict (dict): Names and values of parameters required to run model.

    Returns:
        output_dict: Output from model run.

    """
    np.random.seed(param_dict['rng_seed'])
    model_run = ModelRun(param_dict)
    print(model_run.param_dict['run_id'])
    return model_run.output_dict


class ModelRun:
    """Individual model run.

    Attributes:
        param_dict (dict): Names and values of parameters required to run model.
        population (int): Number of individuals within the modeled community.
        net (nx.Graph): Connected small-world network.
        net_s (float): Measure of network small-worldness.
        hhs (list): Household objects comprising the modeled community.
        stable_mod (float): Contamination modifier associated with the stable behavioral strategy.
        weather_mod (float): Contamination modifier associated with weather-sensitive behavioral strategy.
        stable_hhs (int): Number of households practicing the stable behavioral strategy.
        weather_hhs (int): Number of households practicing the weather-sensitive behavioral strategy.
        env_cycle (int): Environmental cycle ID.
        inf (int): Number of currently infected individuals.
        inf_tot (int): Total number of infections that have occurred.
        time (float): Time elapsed since initiation of model run.

    """

    def __init__(self, param_dict):
        """Sets up and runs the model.

        Args:
            param_dict (dict): Names and values of parameters required to run model.

        """
        self.param_dict = param_dict
        self.population = self.param_dict['hh_n'] * self.param_dict['net_n']
        self.net, self.net_s = self.gen_net()
        self.hhs = self.gen_hhs()
        self.stable_mod = 1
        self.weather_mod = 1 - (np.random.beta(1, self.param_dict['eff_diff']) * 0.9 + 0.05)
        self.weather_hhs = self.set_optimal()
        self.stable_hhs = self.param_dict['net_n'] - self.weather_hhs
        self.env_cycle = 1
        self.inf = 0
        self.inf_tot = 0
        self.time = 0.0
        self.output_dict = {'env_cycle': [],
                            'weather_mod': [],
                            'time': [],
                            'stable_hhs': [],
                            'weather_hhs': [],
                            'inf': [],
                            'inf_tot': []}

        self.run_cycle()

        self.output_dict['run_id'] = self.param_dict['run_id']
        self.output_dict['net_s'] = self.net_s

    def gen_net(self):
        """Generates small-world network.

        Returns:
            net (nx.Graph): Connected small-world network.
            net_s (float): Measure of network small-worldness.

        """
        net = nx.connected_watts_strogatz_graph(self.param_dict['net_n'],
                                                self.param_dict['net_k'],
                                                self.param_dict['net_p'],
                                                seed=self.param_dict['rng_seed'])
        path_length = nx.average_shortest_path_length(net)
        clustering = nx.transitivity(net)
        pl_ref = float(np.log(self.param_dict['net_n']) / np.log(self.param_dict['net_k']))  # reference path length
        c_ref = self.param_dict['net_k'] / self.param_dict['net_n']  # reference clustering
        pl_ratio = path_length / pl_ref
        c_ratio = clustering / c_ref
        net_s = c_ratio / pl_ratio
        return net, net_s

    def gen_hhs(self):
        """Generates interconnected Household objects.

        Returns:
            hhs (list): Household objects comprising the modeled community.

        """
        hhs = [Household(_, self.param_dict['contam'], list(self.net.neighbors(_)), self.param_dict['hh_n'])
               for _ in range(self.param_dict['net_n'])]
        return hhs

    def set_optimal(self):
        """Assigns initially optimal behavioral strategy to randomly selected households and modifies their attributes
        accordingly.

        Returns:
            optimal_init_n (int): Number of households assigned the initially optimal behavioral strategy.

        """
        optimal_init_n = int(self.param_dict['optimal_init'] * self.param_dict['net_n'])
        hhs = range(self.param_dict['net_n'])  # household IDs
        random_hhs = np.random.choice(hhs, optimal_init_n, replace=False)  # randomly selected household IDs
        for hh in random_hhs:
            self.hhs[hh].strat = 'weather_sensitive'
            self.hhs[hh].contam *= self.weather_mod
        return optimal_init_n

    def update_output(self):
        """Updates output dictionary."""
        for attr in self.output_dict.keys():
            self.output_dict[attr].append(getattr(self, attr))

    def run_cycle(self):
        """Runs an environmental cycle."""
        cycle_duration = self.param_dict['duration']
        while self.env_cycle <= self.param_dict['num_cycles']:
            hazards = [hh.gen_hazards(self.param_dict['rec_rate'], self.param_dict['inf_rate']) for hh in self.hhs]
            tot_hazard = np.sum(hazards)
            dt = -np.log(np.random.uniform()) / tot_hazard  # time elapsed since last event
            if self.time + dt > cycle_duration:
                self.update_output()
                self.setup_next_cycle()
                cycle_duration = self.param_dict['duration'] * self.env_cycle
            self.gen_event(dt, hazards, tot_hazard)

    def gen_event(self, dt, hazards, tot_hazard):
        """Generates event through roulette selection from vector of household-level event hazards."""
        self.time += dt
        event_hazards = np.concatenate(np.transpose(hazards) / tot_hazard)
        u = np.random.uniform()
        upper = np.cumsum(event_hazards)  # upper boundaries for roulette selection
        lower = np.insert(upper[0:np.size(event_hazards) - 1], 0, [0])  # lower boundaries for roulette selection
        event, i = divmod(np.where(np.logical_and(np.greater(upper, u), np.less(lower, u)))[0][0],
                          self.param_dict['net_n'])
        hh = self.hhs[i]
        hh.apply_event(event, self.time)
        if event > 0:  # if infection event
            self.resolve_infection(hh)

    def resolve_infection(self, hh):
        """Updates infection and behavioral strategy counters and modifies relevant household attributes following an
        infection event.

        Arguments:
            hh (Household): Household affected by the infection

        """
        self.inf += 1
        self.inf_tot += 1
        neighbors = [self.hhs[neighbor] for neighbor in hh.rels]
        old_strat, new_strat = hh.adapt(neighbors)
        if old_strat != new_strat:
            if old_strat == 'stable':
                self.stable_hhs -= 1
                self.weather_hhs += 1
                hh.contam = self.param_dict['contam'] * self.weather_mod
            else:  # 'weather-sensitive'
                self.stable_hhs += 1
                self.weather_hhs -= 1
                hh.contam = self.param_dict['contam'] * self.stable_mod

    def setup_next_cycle(self):
        """Sets up the next environmental cycle by changing the exposure modifier associated with the weather-sensitive
        behavioral strategy and resetting relevant attributes."""
        self.env_cycle += 1
        self.inf = 0
        if self.param_dict['scenario'] == 'punctuated':
            self.weather_mod = 1 + (np.random.beta(1, self.param_dict['eff_diff']) * 0.9 + 0.05)
        else:  # 'baseline'
            self.weather_mod = 1 - (np.random.beta(1, self.param_dict['eff_diff']) * 0.9 + 0.05)
        for hh in self.hhs:
            hh.contam = self.param_dict['contam'] * self.weather_mod if hh.strat == 'weather_sensitive' else hh.contam


class Household:
    """Household within the modeled community.

    Attributes:
        hh_id (int): Household identifier.
        contam (float): Contamination level to which household is exposed.
        rels (list): IDs of household's network neighbors (relations).
        sus (int): Number of susceptible individuals.
        inf (int): Number of infected individuals.
        rec (int): Number of individuals who have recovered.
        strat (str): Household's behavioral strategy.
        last_inf (float): Time of last infection event.
        inf_intervals (list): Time intervals between infection events.

    """

    def __init__(self, hh_id, contam, rels, sus):
        self.hh_id = hh_id
        self.contam = contam
        self.rels = rels
        self.sus = sus
        self.inf = 0
        self.rec = 0
        self.strat = 'stable'
        self.last_inf = 0.0
        self.inf_intervals = []

    def gen_hazards(self, rec_rate, inf_rate):
        """Generates event hazards.

        Returns:
            rec_hazard: Household's recovery hazard.
            inf_hazard: Household's infection hazard.

        """
        rec_hazard = self.inf * rec_rate
        inf_hazard = self.sus * self.contam * inf_rate
        return rec_hazard, inf_hazard

    def apply_event(self, event, time):
        """Modifies household attributes to reflect infection or recovery event.

        Args:
            event (int): Event type (recovery or infection).
            time (float): Event time.

        """
        if event == 0:
            self.inf -= 1
            self.sus += 1
        else:
            self.inf += 1
            self.sus -= 1
            self.inf_intervals.append(time - self.last_inf)
            self.last_inf = time

    def adapt(self, neighbors):
        """Updates household's behavioral strategy by randomly selecting from the strategies of network neighbors that
        have experienced longer mean intervals between infections.

        Args:
            neighbors (list): Household's network neighbors (Household objects).

        Returns:
            old_strat (str): Household's previous behavioral strategy.
            self.strat (str): Household's newly selected behavioral strategy.

        """
        old_strat = self.strat
        mean_inf_interval = np.mean(self.inf_intervals)  # mean interval between infections
        neighbor_strats = []  # behavioral strategies of better-off neighbors
        for neighbor in neighbors:
            if not neighbor.inf_intervals or np.mean(neighbor.inf_intervals) > mean_inf_interval:
                neighbor_strats.append(neighbor.strat)
        if neighbor_strats:
            self.strat = np.random.choice(neighbor_strats)
        return old_strat, self.strat


if __name__ == '__main__':
    main()

__version__ = '0.5.5'
__date__ = '12202018'
