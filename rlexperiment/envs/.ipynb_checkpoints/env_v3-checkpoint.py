# Simplified environment, performance optimized version, v3, accessibility calculation with np, and population-weighted calculation target.
from stable_baselines3 import PPO
import gymnasium as gym
import time
import numpy as np
import os
from geopy.distance import geodesic
from rlexperiment.utils.spatialAccessibility import calculate_accessibility_use_np, calculate_gini
import pandas as pd
from datetime import datetime

class HospitalEnvironment_enhanced_v3(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        # Call the constructor of the parent class
        super(HospitalEnvironment_enhanced_v3, self).__init__()
        self.config = config


        # Define observation space and action space
        self.observation_space = config.get("observation_space")
        self.action_space = config.get("action_space")

        # get params form config file
        self.real_world = config.get("real_world", False)  # determine whether it is a real-world environment
        self.TwoSFCA_beta = config.get("the_2SFCA_beta",1)
        print(f"The value of TwoSFCA_beta is: {self.TwoSFCA_beta}")
        # time.sleep(5)
        self.use_population_weighted = config.get("use_population_weighted",True)
        self.time_log = [] # time log for performance evaluation
        self.model_path = config.get("model_path", None)  # get the model path
        self.model = None  # init the model to None
        # init the environment state
        self._initial_state(config)
        # if get the model path, load model, skip training
        if self.model_path:
            self.load_model(self.model_path)

    def load_model(self, model_path):
        """
        load the trained PPO model 
        """
        print(f"Loading Model: {model_path}")
        self.model = PPO.load(model_path)   # use stable baselines3 to load PPO model
        self.model.set_env(self)  # Set the environment to ensure that the model can run in the current environment.
        self.model.policy.eval()  # Set to evaluation mode, without gradient calculation.

    def _initial_state(self, config):
        
        start_time = time.time()  # record the start time of the environment
        # initlize the environment state
        if self.real_world:
            self.total_distance = 0
            self.transfers_executed = 0
            self.accessibility_cap = config.get("accessibility_cap", 0.8) # Andy - means that the agent must leave at least this amount of supply per action
            self.hard_acc_cap = config.get("hard_acc_cap", 0.8) # Andy
            self.hard_cap = config.get("hard_cap", False)
            self.research_area = config.get("research_area","research_area")
            self.df_Destinations = config.get("df_Destinations", None).copy()  # the data frame of destinations
            self.df_Origins = config.get("df_Origins", None).copy()  # the data frame of origins

            
            self.df_od_matrix = config.get("df_od_matrix", None).copy()  # od matrix of the origins and destinations
            self.reward_expander = config.get("reward_expander", 100000) # Andy - changed to 100x, 1000000 for the mean reward of 10k
            self.var_expander = config.get("var_expander",10000000)
            self.remain_origin_supply = config.get("remain_origin_supply",True)
            self.max_distance = config.get("max_distance",10)
            self.qp_like = config.get("qp_like",False)
            # the qp_like parameter is used to close the two switches below
            if self.qp_like:
                self.remain_origin_supply = False
                self.max_distance = 0 # here we set the max_distance to 0, so that the max_distance constraint is not used.
            self.max_steps_per_experiment = config.get("max_steps_per_experiment",200)
            self.Punishment_for_Violating_Conditions = config.get("Punishment_for_Violating_Conditions",10000)
            # if in real world environment, check whether the destination and origin data are provided in config
            if self.df_Destinations is None or self.df_Origins is None:
                raise ValueError("Real world environment requires gdf_Destination and gdf_Origin to be provided in config.")

            # use the provided function to calculate accessibility and update destination and origin data
            self.df_Origins = calculate_accessibility_use_np(self.df_Destinations, self.df_Origins, self.df_od_matrix, beta=self.TwoSFCA_beta,print_out=False)
            
            # self.df_Origins = calculate_gini(self.df_Destinations, self.df_Origins, self.df_od_matrix)
            
            # look at the summary of the origin data
            origins_summary = self.df_Origins.describe()

            # print("Origin Data Summary:")
            # print(origins_summary)
            # print("*"*66)
            # time.sleep(2)
            
            # initialize destinations and origins from geodataframe
            self.Destinations_dict = self._initialize_Destinations_from_geodataframe()
            self.total_supply = self.df_Destinations['D_Supply'].sum()  # 总供给
            self.Origins_dict = self._initialize_Origins_from_geodataframe()
            self.capped_supply = (self.df_Destinations['D_Supply'] * self.hard_acc_cap).set_axis(self.df_Destinations['DestinationID'])


            # calculate the standard deviation of accessibility of origins (e.g. residential area)
            self.accessibility_scores = [Origin['accessibility'] for Origin in self.Origins_dict.values()]
            # print(self.Origins_dict)
            self.min_accessibility_standard_deviation = np.std(self.accessibility_scores, ddof=1)
            self.min_weighted_acc_var,self.min_acc_var = self._weighted_variance_supply_demand(self.df_Origins, value_col='accessibility', weight_col='O_Demand')
            self.initial_weighted_acc_var = self.min_weighted_acc_var
            self.initial_min_acc_var = self.min_acc_var


            # get the traget ratio and calculate the target value of accessibility variance
            self.target_ratio = config.get("target_ratio", 0.4)
            self.accessibility_variance_target_cal = self.target_ratio * self.initial_min_acc_var
            self.accessibility_weighted_variance_target_cal = self.target_ratio * self.initial_weighted_acc_var

            self.areas_need_focus = config.get("areas_need_focus",None)
            if self.areas_need_focus:

              # filter the accessibility scores of the specified areas
              self.accessibility_scores_for_focus_areas = [
                  self.Origins_dict[area_id]['accessibility'] for area_id in self.areas_need_focus
                  if area_id in self.Origins_dict
              ]
              # calculate the average accessibility of the specified areas
              if self.accessibility_scores_for_focus_areas:
                  self.average_accessibility_of_focus_areas = np.mean(self.accessibility_scores_for_focus_areas)
                  self.init_average_accessibility_of_focus_areas = np.mean(self.accessibility_scores_for_focus_areas)
                  self.max_average_accessibility_of_focus_areas = self.init_average_accessibility_of_focus_areas
                  print(f'Init avg value of accessibility of the target area: {self.init_average_accessibility_of_focus_areas}')

              else:
                  average_accessibility_of_focus_areas = None
                  print("No accessibility scores found for the specified focus areas.")

            # initialize the start and end destination id variables
            self.start_Destination_id_record = None
            self.end_Destination_id_record = None
            # record the number of steps since the last reset
            self.step_since_reset = 0

            # print the initialization information
            print(f'Success: Creating supply point and request point from gdf')
            print(f'{"Init accessibility variance:":<40}{self.initial_min_acc_var:<40}')
            print(f'{"Init accessibility weighted variance:":<40}{self.initial_weighted_acc_var:<40}')
            print(f'{"Variance target =":<40}{self.accessibility_variance_target_cal:<40}')
            print(f'{"Weighted Variance target =":<40}{self.accessibility_weighted_variance_target_cal:<40}')
            print(f'{"target_ratio =":<40}{self.target_ratio:<40}')
            # print(self.time_log)
        else:
            # if not in the real-world environment, initialize the virtual destinations and origins
            self.Destinations = self._initialize_Destinations(num_Destinations)
            self.Origins = self._initialize_Origins(num_Origins)
        end_time = time.time()  # record the end time of the environment
        self.time_log.append(("initial_state", end_time - start_time))
        print(f'{"time of init is":<40} {end_time - start_time}')
        print('Env Init Finished'+'#'*66)

    def _weighted_variance_supply_demand(self, df_origins, value_col='accessibility', weight_col='O_Demand'):
        """
        Calculate the cumulative value of D_i * (A_i - a)^2, as well as the ordinary variance.
        Parameters:
        df_origins (DataFrame): A DataFrame containing demand point data, which must include 'accessibility' and 'O_Demand' columns.
        value_col (str): The name of the accessibility column, defaults to 'accessibility'.
        weight_col (str): The name of the demand column, defaults to 'O_Demand'.

        Returns:
        tuple: A tuple containing the cumulative value of D_i * (A_i - a)^2 (weighted variance) and the ordinary variance.
        """

        # extract accessibility and demand values
        values = df_origins[value_col].values
        weights = df_origins[weight_col].values
        # create a DataFrame
        temp_df = pd.DataFrame({
            'values': values,
            'weights': weights
        })
        # summary = temp_df.describe()
        # print("summary:")
        # print(summary)
        # print('*'*66)
        # time.sleep(2)

        # calculate total demand
        total_demand = weights.sum()

        # calculate a = total_supply / total_demand
        a = self.total_supply / total_demand


        # caculating weighted_acc_var
        weighted_acc_var = np.sum(weights * ((values - a) ** 2)) / total_demand
        # calulate the variance of accessibility
        regular_variance = np.var(values, ddof=1)

        print(f'{"regular_variance is":<40} {regular_variance}')
        print(f'{"weighted_acc_var is":<40} {weighted_acc_var}')

        return weighted_acc_var, regular_variance

    def step_time_logger(self,label,input_time,print_out = False):
        temp_time = time.time()
        self.step_time_log.append((label,temp_time-input_time))
        # print time information
        if print_out:
            print(f'{"time of":<15} {label:<25} {"is":<5} {temp_time - input_time:<10}')
            return temp_time
        else:
            return temp_time

    def _initialize_Destinations(self, num_Destinations):
        # randomly initialize supply point (e.g. hospital) information
        return [self._random_Destination() for _ in range(num_Destinations)]
    def _initialize_Origins(self, num_Origins):
        # randomly initialize demand point (e.g. residential area) information
        return [self._random_Originial_area() for _ in range(num_Origins)]

    def _initialize_Destinations_from_geodataframe(self):
        start_time = time.time()  # record the start time of the environment
        
        # initlize the destinations(supply points, e.g. hospital) from geodataframe
        selected_columns = ['DestinationID', 'lng', 'lat', 'D_Supply']
        df_selected = self.df_Destinations[selected_columns]
        result_dict = df_selected.set_index('DestinationID').to_dict(orient='index')

        # make sure the dict structure is as expected
        result_dict = {
            DestinationID: {
                'DestinationID': DestinationID,
                'lng': row['lng'],
                'lat': row['lat'],
                'D_Supply': row['D_Supply']
            }
            for DestinationID, row in result_dict.items()
        }

        end_time = time.time()  # record the end time
        self.time_log.append(("_initialize_Destinations_from_geodataframe", end_time - start_time))
        #print(f'time of _initialize_Destinations_from_geodataframe is {end_time - start_time}')

        return result_dict


    def _initialize_Origins_from_geodataframe(self):
      start_time = time.time()  # # record the start time

      
      # access the data using column names to avoid the overhead of iterating over rows
      result_dict = self.df_Origins.set_index('OriginID').to_dict(orient='index')

      end_time = time.time()
      self.time_log.append(("_initialize_Origins_from_geodataframe", end_time - start_time))
      #print(f'time of _initialize_Origins_from_geodataframe is {end_time - start_time}')

      return result_dict

    def _update_accessibility_from_geodataframe(self):
      start_time = time.time()  # # record the start time

      # create a dict to quickly look up accessibility values
      accessibility_map = self.df_Origins.set_index('OriginID')['accessibility'].to_dict()

      # update the accessibility attribute for each origin using the map function
      for origin_id in self.Origins_dict.keys():
          if origin_id in accessibility_map:
              self.Origins_dict[origin_id]['accessibility'] = accessibility_map[origin_id]

      self.accessibility_scores = [Origin['accessibility'] for Origin in self.Origins_dict.values()]
      self.accessibility_std = np.std(self.accessibility_scores,ddof=1) # std
      self.accessibility_var = np.var(self.accessibility_scores, ddof=1) # variance
      print(f'current std {self.accessibility_std}')
      print(f'current accessibility var {self.accessibility_var}')

      end_time = time.time()
      self.time_log.append(("update_accessibility_from_geodataframe", end_time - start_time))
      #print(f'time of update_accessibility_from_geodataframe is {end_time - start_time}')

    def _random_Destination(self):
        # randomly generate a supply point (e.g. hospital) state
        lng = np.random.uniform(self.observation_space.low[0], self.observation_space.high[0])
        lat = np.random.uniform(self.observation_space.low[1], self.observation_space.high[1])
        D_Supply = np.random.randint(self.observation_space.low[3], self.observation_space.high[3] + 1)
        return {'DestinationID': 0, 'lng': lng, 'lat': lat, 'D_Supply': D_Supply}

    def _random_Originial_area(self):
        # randomly generate an origin (e.g. residential area) state
        lng = np.random.uniform(self.observation_space.low[0], self.observation_space.high[0])
        lat = np.random.uniform(self.observation_space.low[1], self.observation_space.high[1])
        O_Demand = np.random.randint(self.observation_space.low[2], self.observation_space.high[2] + 1)
        accessibility = np.random.uniform(0, 100)
        return {'lng': lng, 'lat': lat, 'O_Demand': O_Demand, 'accessibility': accessibility}

    def get_state(self):
        # Implement the logic to get the current state
        # The state of each supply point (e.g., hospital) and demand point (e.g., residential area) can be represented by the following attributes:
        # Supply point (e.g., hospital): (ID, Longitude, Latitude, Total Labor Force)
        # Demand point (e.g., residential area): (Longitude, Latitude, Population, Accessibility)

        start_time = time.time()  # record the start time
        Destinations_state = [(Destination['DestinationID'], Destination['lng'], Destination['lat'], Destination['D_Supply']) for Destination in self.Destinations_dict.values()]
        Origins_state = [(area['lng'], area['lat'], area['O_Demand'], area['accessibility']) for area in self.Origins_dict.values()]
        state = {
            'Destinations': Destinations_state,
            'Origins': Origins_state
        }
        end_time = time.time()  # record the end time
        self.time_log.append(("get_state", end_time - start_time))
        #print(f'time of get_state is {end_time - start_time}')


        return state

    def _get_observation(self):
        start_time = time.time()  # record the start time
        state = self.get_state()
        # convert the state dictionary into a list
        Destinations_data = [substate for Destination in state['Destinations'] for substate in Destination]
        Originial_data = [substate for area in state['Origins'] for substate in area]

        # merge the supply point (e.g., hospital) and demand point (e.g., residential area) data into a single list
        flat_state = Destinations_data + Originial_data
        # print(f"observate flat_state{flat_state}")
        end_time = time.time()  # record the end time
        self.time_log.append(("_get_observation", end_time - start_time))
        #print(f'time of _get_observation is {end_time - start_time}')

        return np.array(flat_state, dtype=np.float32)

    def reset(self, seed=None):
        start_time = time.time()  # record the start time
        if seed is not None:
            np.random.seed(seed)
        # data recorder
        data_recorder = True
        # data_recorder = False

        if data_recorder:
            # get current time and convert it to a string, e.g. '2024-07-09_15-30-00'
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # print(self.Destinations_dict)
            # convert the dictionary to a DataFrame
            df_Destinations_result_output = pd.DataFrame(list(self.Destinations_dict.values()))
            df_Origins_result_output = pd.DataFrame(list(self.Origins_dict.values()))
            # export
            df_Destinations_result_output.to_csv(f'./Output/{self.research_area}_{self.TwoSFCA_beta}_Destinations_{current_time}.csv', index=False)
            df_Origins_result_output.to_csv(f'./Output/{self.research_area}_{self.TwoSFCA_beta}_Origins_{current_time}.csv', index=False)
            print(f'result saved to ./Output/{self.research_area}_{self.TwoSFCA_beta}_Destinations_{current_time}.csv')
            print(f'result saved to ./Output/{self.research_area}_{self.TwoSFCA_beta}_Origins_{current_time}.csv')
            print(f'Results Saved: {current_time}')
        else:
            print('Current result csv not saved')
        
        # review the last step
        print(f"Inititial weighted variance:{self.initial_weighted_acc_var} -> Minimal Weighted Variance {self.min_weighted_acc_var}")
        print(f"Optimization ratio of weighted variance:{(self.min_weighted_acc_var/self.initial_weighted_acc_var)*100}%")
        # 
        print(f'Data Resetting')

        self.df_Destinations = self.config.get("df_Destinations", None)
        
        self.df_Origins = self.config.get("df_Origins", None)

        self._initial_state(self.config)
        print('Reset Done'+'*'*66)
        observation = self._get_observation()
        end_time = time.time()  # record the end time
        self.time_log.append(("reset", end_time - start_time))
        #print(f'time of reset is {end_time - start_time}')
        # print(self.time_log) # warning: this might print too much
        self.time_log = []  # reset the time log

        # return the observation and additional information (here we use None because there is no additional information)
        return observation, None
    import numpy as np

    def calculate_average_accessibility_of_focused_areas(self):
        if self.areas_need_focus:
            # fliter and get the accessibility scores of the specified areas
            self.accessibility_scores_for_focus_areas = [
                self.Origins_dict[area_id]['accessibility'] for area_id in self.areas_need_focus
                if area_id in self.Origins_dict
            ]

            # calculate the average accessibility of the specified areas
            if self.accessibility_scores_for_focus_areas:
                self.average_accessibility_of_focus_areas = np.mean(self.accessibility_scores_for_focus_areas)
            else:
                self.average_accessibility_of_focus_areas = None
                print("No accessibility scores found for the specified focus areas.")
        else:
            self.average_accessibility_of_focus_areas = None
            print("No focus areas specified.")
    
    def check_valid_action(self, start_Destination_id, end_Destination_id):
        """
        Check if an action is valid and decide whether to execute it based on conditions.

        Parameters:
        start_Destination_id (int): The ID of the starting destination.
        end_Destination_id (int): The ID of the ending destination.

        Returns:
        tuple: A tuple containing two elements.
            - The first element is a boolean indicating whether the action is valid.
            - The second element is a penalty value, which is the penalty if the action is invalid, otherwise 0.

        Logic:
        1. If `start_Destination_id` and `end_Destination_id` are the same as the recorded `start_Destination_id_record` and `end_Destination_id_record`,
        then the action is considered invalid, and `False` and the penalty value `Punishment_for_Violating_Conditions` are returned.
        2. Otherwise, update the recorded `start_Destination_id_record` and `end_Destination_id_record`, and return `True` and 0.
        """
        if start_Destination_id == self.start_Destination_id_record and end_Destination_id == self.end_Destination_id_record:
            # if the start and end destination IDs match the recorded ones, then the action is invalid

            print("\033[91m punishment done, no change of current status, nothing executed. Because start_Destination_id and end_Destination_id match record simultaneously \033[0m")
            return False, self.Punishment_for_Violating_Conditions
        
        # update the recorded `start_Destination_id_record` and `end_Destination_id_record`
        self.start_Destination_id_record = start_Destination_id
        self.end_Destination_id_record = end_Destination_id
        
        # if the action is valid, return True and 0
        return True, 0
    def check_destinations_existence(self, start_Destination_id, end_Destination_id):
        if start_Destination_id not in self.Destinations_dict or end_Destination_id not in self.Destinations_dict:
            print('\033[91m cross border, action invalidated \033[0m')
            return False, self.Punishment_for_Violating_Conditions
        return True, 0

    
    def check_transfer_distance(self, start_Destination, end_Destination):
        if self.qp_like:
            return True, 0  # do not need to calucate the distance
        elif self.max_distance > 0:
            distance = geodesic(
                (start_Destination['lat'], start_Destination['lng']),
                (end_Destination['lat'], end_Destination['lng'])
            ).kilometers
            if distance > self.max_distance:
                print(f'\033[91m distant to supply point {distance}km, action invalidated\033[0m')
                return False, self.Punishment_for_Violating_Conditions
        return True, 0

    def check_supply_availability(self, start_Destination, num_transfer):
        if start_Destination['D_Supply'] < num_transfer:
            if self.remain_origin_supply:
                print('\033[91m Currently Qp mode, too many people in transfer, become total number of the origin. \033[0m')
                num_transfer = start_Destination['D_Supply']
            else:
                print('\033[91m too many transfer in one go, action invalidated \033[0m')
                return False, num_transfer, self.Punishment_for_Violating_Conditions
        return True, num_transfer, 0

    
    def update_destinations(self, start_Destination, end_Destination, num_transfer):
        start_Destination['D_Supply'] -= num_transfer
        end_Destination['D_Supply'] += num_transfer
        print('Labor transfer done')
        
        start_Destination_row = self.df_Destinations[self.df_Destinations['DestinationID'] == start_Destination['DestinationID']]
        end_Destination_row = self.df_Destinations[self.df_Destinations['DestinationID'] == end_Destination['DestinationID']]
        
        if not start_Destination_row.empty and not end_Destination_row.empty:
            self.df_Destinations.at[start_Destination_row.index[0], 'D_Supply'] = start_Destination['D_Supply']
            self.df_Destinations.at[end_Destination_row.index[0], 'D_Supply'] = end_Destination['D_Supply']
        else:
            print('\033[91m No corresponding supply point found, no update, execute punishment.\033[0m')
            return False, self.Punishment_for_Violating_Conditions
        return True, 0

    # not used
    def calculate_reward(self):
        reward = 0
        # update weighted accessibility variance and check if it reaches the optimal value
        self.weighted_acc_var, self.acc_var = self._weighted_variance_supply_demand(self.df_Origins, value_col='accessibility', weight_col='O_Demand')
        if self.weighted_acc_var < self.min_weighted_acc_var:
            delta = self.min_weighted_acc_var - self.weighted_acc_var
            reward += delta * self.var_expander
            self.min_weighted_acc_var = self.weighted_acc_var

        # the reward for the average accessibility of the specified areas
        if self.areas_need_focus:
            self.calculate_average_accessibility_of_focused_areas()
            if self.average_accessibility_of_focus_areas > self.max_average_accessibility_of_focus_areas:
                delta = self.average_accessibility_of_focus_areas - self.max_average_accessibility_of_focus_areas
                reward += delta * self.reward_expander
                self.max_average_accessibility_of_focus_areas = self.average_accessibility_of_focus_areas
        
        return reward



    # take action and update the environment state
    def step(self, action):
        self.step_time_log = []   # create an undefined list for recording time
        start_time = time.time()  # record the start time
        step_start_time = time.time()


        # determine if the task is completed
        reward = 0 # initialize the reward
        done = False  # suppose it will never be completed
        truncated = False
        info = {}# here is an empty dictionary for storing additional information

        # decode the action
        start_Destination_id, end_Destination_id, num_transfer = action
        temp_time = self.step_time_logger("Action Analysis", step_start_time)
        # action marker
        ready_to_take_action = True
        print(f'This round No.{self.step_since_reset} step,New step generated \namount of supply:{num_transfer} \nfrom (Supply_Origin_ID) {start_Destination_id}, to (Supply_Destination_id):{end_Destination_id} \n start checking legality next ')
        # step counter
        self.step_since_reset += 1

        # to check if the action is valid
        valid_action, penalty = self.check_valid_action(start_Destination_id, end_Destination_id)
        if not valid_action:
            reward -= penalty
            done = True
            num_transfer = 0
            return self._get_observation(), reward, done, truncated, info

        temp_time = self.step_time_logger(" check if starting id and ending id match", temp_time)

        if self.step_since_reset > self.max_steps_per_experiment:
            print('Current step reach limit. No action, no punishment, no status update.')
            done = True
            # reward  -= self.Punishment_for_Violating_Conditions  # 给予负奖励
            return self._get_observation(), reward, done, truncated, info


        # check if the number of transfer is valid
        valid_destinations, penalty = self.check_destinations_existence(start_Destination_id, end_Destination_id)
        if not valid_destinations:
            reward -= penalty
            num_transfer = 0
            return self._get_observation(), reward, done, truncated, info
        temp_time=self.step_time_logger(" check if action cross border", temp_time)

        start_Destination = self.Destinations_dict.get(start_Destination_id)
        end_Destination = self.Destinations_dict.get(end_Destination_id)

        # print(f' check id: start_Destination:{start_Destination},start_Destination_id:{start_Destination_id}')
        # print(f' check id: end_Destination:{end_Destination},end_Destination_id:{end_Destination_id}')

        if start_Destination is not None and end_Destination is not None:
            # only print when both start_Destination and end_Destination are not None
            print(f' check legality of the move {num_transfer} doc, from {start_Destination["DestinationID"]} (with D_Supply:{start_Destination["D_Supply"]}) to {end_Destination["DestinationID"]} (with D_Supply:{end_Destination["D_Supply"]}) check effectiveness of the move')

        else:
            if start_Destination is None:
                print(f'Staring supply point not found, ID of {start_Destination_id} supply point information')
                reward -= self.Punishment_for_Violating_Conditions
                num_transfer = 0
                ready_to_take_action = False
                print('\033[91m cross border, invalidate action\033[0m')
                return self._get_observation(), reward, done, truncated, info

            if end_Destination is None:
                reward -= self.Punishment_for_Violating_Conditions
                num_transfer = 0
                ready_to_take_action = False
                print(f'Staring supply point not found, ID of  {end_Destination_id} supply point information')
                return self._get_observation(), reward, done, truncated, info
        
        temp_time=self.step_time_logger(" check whether staring and ending points have been defined or not.", temp_time)

        if start_Destination is None or end_Destination is None:
            # raise ValueError("Invalid action: start or end Destination not found.")
            reward -= self.Punishment_for_Violating_Conditions
            num_transfer = 0

            ready_to_take_action = False
            print('\033[91m cross border, invalidate action\033[0m')


        # check the supply availability
        valid_supply, num_transfer, penalty = self.check_supply_availability(start_Destination, num_transfer)
        if not valid_supply:
            reward -= penalty
            return self._get_observation(), reward, done, truncated, info
        temp_time=self.step_time_logger(" check if transferring too many people in one go", temp_time)

        # # Check if agent is taking away too much
        # if num_transfer > start_Destination["D_Supply"] * (1 - self.accessibility_cap):
        #     reward -= penalty
        #     return self._get_observation(), reward, done, truncated, info

        if self.hard_cap: # Check if the supply is being reduced too much
            if start_Destination['D_Supply'] - num_transfer < self.capped_supply[start_Destination['DestinationID']]:
                reward -= penalty
                return self._get_observation(), reward, done, truncated, info

        print('transfer distance check ')

        if self.qp_like:
            print('Currently qp mode, no need to calculate distance.')
        elif self.max_distance>0:
            print('Moving distant limit met.')
            # caluculate the distance between the starting and ending supply points
            distance = geodesic(
                (start_Destination['lat'], start_Destination['lng']),
                (end_Destination['lat'], end_Destination['lng'])
            ).kilometers
            # if too far, then do not execute the action
            if distance > self.max_distance:
                reward -= self.Punishment_for_Violating_Conditions
                num_transfer = 0
                ready_to_take_action = False
                print(f'\033[91m Distance between two supply points {distance}km,Invalidate Action \033[0m ')
        else:
            print('No transfer distance limit')

        # Andy - Check 
            

        if ready_to_take_action:
            print('Action Valid \n Start transfer labor')
            self.transfers_executed += 1
            self.total_distance += distance

            print(f' Current total distance traveled: {self.total_distance} \nCurrent valid actions executed: {self.transfers_executed}')
            start_Destination['D_Supply'] -= num_transfer
            end_Destination['D_Supply'] += num_transfer
            print('Labor transfer done.')
            temp_time=self.step_time_logger("Labor transfer (dict)", temp_time)

            # use DestinationID to locate the corresponding row
            start_Destination_row = self.df_Destinations[self.df_Destinations['DestinationID'] == start_Destination_id]
            end_Destination_row = self.df_Destinations[self.df_Destinations['DestinationID'] == end_Destination_id]
            if not start_Destination_row.empty and not end_Destination_row.empty: # 这里是为了检测
                self.df_Destinations.at[start_Destination_row.index[0], 'D_Supply'] = start_Destination['D_Supply']
                self.df_Destinations.at[end_Destination_row.index[0], 'D_Supply'] = end_Destination['D_Supply']
            else:
                print('\033[91m Supply point not found. No update. Execute punishment. No current status update.\033[0m')
                reward -= self.Punishment_for_Violating_Conditions
                done = True
                return self._get_observation(), reward, done, truncated, info

            temp_time=self.step_time_logger("Update destination的df", temp_time)

            # print(f'after transfer, supply point number: {len(self.df_Destinations)}')
            # Andy - gini
            self.df_Origins = calculate_accessibility_use_np(self.df_Destinations, self.df_Origins,self.df_od_matrix,beta=self.TwoSFCA_beta,print_out=False)
            # self.df_Origins = calculate_gini(self.df_Destinations, self.df_Origins, self.df_od_matrix)
            temp_time=self.step_time_logger("Calculate accessibility", temp_time)

            # update the states
            self._update_accessibility_from_geodataframe()
            temp_time=self.step_time_logger("Update env status", temp_time)

            # check if the weighted accessibility variance is lower than the minimum value variance  1e-5 * 1000000 = 1
            self.weighted_acc_var,self.acc_var = self._weighted_variance_supply_demand(self.df_Origins, value_col='accessibility', weight_col='O_Demand')
            if self.weighted_acc_var < self.min_weighted_acc_var:
                # calculate the difference between the current and the minimum weighted accessibility variance and take it as the reward
                delta = self.min_weighted_acc_var - self.weighted_acc_var
                reward = delta*self.var_expander
                print(f'weighted_acc_var from {self.min_weighted_acc_var} to {self.weighted_acc_var}')
                print(f'Requirement weighted variance of accessibility {self.weighted_acc_var}')
                print(f'previous ratio is {(self.min_weighted_acc_var/self.initial_weighted_acc_var)*100}%')
                print(f'current ratio is {(self.weighted_acc_var/self.initial_weighted_acc_var)*100}%')

                # update the minimum weighted accessibility variance
                self.min_weighted_acc_var = self.weighted_acc_var
                print('Minimal weighted variance of accessibility updated.')

                print(f'\033[33m reward + {reward}\033[0m')  # print the reward in yellow
            
            # # caluculate the average accessibility of the specified areas 0.02 -> 0.03 .01 * 1000000 = 1000
            # self.calculate_average_accessibility_of_focused_areas()
            # if self.areas_need_focus:
            #   if self.average_accessibility_of_focus_areas > self.max_average_accessibility_of_focus_areas:
            #     delta = self.average_accessibility_of_focus_areas - self.max_average_accessibility_of_focus_areas
            #     # reward the agent
            #     reward += delta*self.reward_expander
            #     print(f'average_accessibility_of_focus_areas from {self.max_average_accessibility_of_focus_areas} to {self.average_accessibility_of_focus_areas}')
            #     print(f'Avg accessibility of the target area:{self.average_accessibility_of_focus_areas}')
            #     print(f'previous ratio is {(self.max_average_accessibility_of_focus_areas/self.init_average_accessibility_of_focus_areas)*100}%')
            #     print(f'current ratio is {(self.average_accessibility_of_focus_areas/self.init_average_accessibility_of_focus_areas)*100}%')

            #     # update the maximum average accessibility of the specified areas
            #     self.max_average_accessibility_of_focus_areas = self.average_accessibility_of_focus_areas

            #     print(f'\033[33m reward + {reward}\033[0m')  # print the reward in yellow

            # Andy Qin Cut the reward based on reward and on accessibility decrease
            # reward -= 0.0001 * distance
            print(f'Cut the reward with a distance of {distance}')

            
            temp_time=self.step_time_logger("Calculate avg accessibility of the target area", temp_time)


            # check if the weighted accessibility variance is lower than the target value
            if self.weighted_acc_var<=self.accessibility_weighted_variance_target_cal:
                print(f'\033[33m goal of weighted variance of accessibility reached {self.weighted_acc_var},less than target {self.accessibility_weighted_variance_target_cal}\033[0m')
                print(f'current ratio is {(self.weighted_acc_var/self.initial_weighted_acc_var)*100}%')
                reward+=100000
                done = True

        # get new observation
        print('New Status')
        observation = self._get_observation()
        temp_time=self.step_time_logger("Observe status after update", temp_time)

        end_time = time.time()  # record the end time
        self.time_log.append(("step", end_time - start_time))
        #print(f'time of step is {end_time - start_time}')

        # return a tuple include : observation, reward, done, truncated, info
        # print(f"{'Observation:':<40} {observation}, \n{'Reward:' :<40}{reward}, \nDone: {done}, \nTruncated:{truncated},\nInfo: {info}")
        # print(f"{'Observation:':<40} {observation}, \n{'Reward:':<40} {reward}, \n{'Done:':<40} {done}, \n{'Truncated:':<40} {truncated}, \n{'Info:':<40} {info}")
        print(f"{{'Reward:':<40}} {reward}, \n{{'Done:':<40}} {done}, \n{{'Truncated:':<40}} {truncated}, \n{{'Info:':<40}} {info}")
        # print(f"step time:：\n{self.step_time_log}")
        
        # check if observation contains None
        contains_none = any(item is None for item in observation)
        if contains_none:
            print("Observation None datatype in list")
        else:
            print("Observation no None datatype in list")

        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        # render the environment
        pass