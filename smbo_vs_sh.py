import ConfigSpace
import numpy as np
import argparse

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import matplotlib.pyplot as plt
from surrogate_model import SurrogateModel
from operator import itemgetter

import math
#can we change the value of 1 in the first question
#what are the names of the datasets
import ConfigSpace
from ConfigSpace import Constant
import numpy as np
import typing
import random
#from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import math
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from random_search import RandomSearch
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from surrogate_model import SurrogateModel

def identify_categorical_numerical( df):
        categorical_cols = []
        numerical_cols = []

        for col in df.columns:
            # If the column has an object data type or boolean (which could represent categories), treat as categorical
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                categorical_cols.append(col)
            # Otherwise, it's treated as numerical
            else:
                numerical_cols.append(col)
        
        return categorical_cols, numerical_cols
    
    
    


def clean_configuration(config_space, theta):
    default_values = {}

    # Get the default value
    
    for key, _ in config_space.items():
        default_values[key] = config_space.get_hyperparameter(key).default_value
    
    
    # Step 2: Find missing keys in the partial configuration
    missing_keys = [key for key in default_values.keys() if key not in theta]
    
    # Step 3: Add missing keys with default values to the configuration
    for key in missing_keys:
        theta[key] = default_values[key]
        
    return theta


class SequentialModelBasedOptimization(object):

    def __init__(self, config_space, max_anchor, ex_surrogate_class, sample_space, tracker_indexes):
        """
        Initializes empty variables for the model, the list of runs (capital R), and the incumbent
        (theta_inc being the best found hyperparameters, theta_inc_performance being the performance
        associated with it)
        """
        self.config_space = config_space
        self.R = None
        self.theta_inc =None
        self.theta_inc_performance = None
        self.internal_surrogate_model = None
        self.result_performance = []
        self.random_search = RandomSearch(self.config_space)
        self.max_anchor = max_anchor
        self.all_performances = []
        self.ex_surrogate_class = ex_surrogate_class
        self.tracker_indexes = tracker_indexes
        self.sample_space = sample_space
        
    def initialize(self, capital_phi: typing.List[typing.Tuple[typing.Dict, float]]) -> None:
        """
        Initializes the model with a set of initial configurations, before it can make recommendations
        which configurations are in good regions. Note that we are minimising (lower values are preferred)

        :param capital_phi: a list of tuples, each tuple being a configuration and the performance (typically,
        error rate)
        """
        """
        receive sample configuration and respective performance using the ConfigSpace library for a given configuration 
        
        """
        self.R = capital_phi
        
        print("initilised the model")
        
        best_value = min(capital_phi,key = lambda x: x[1])
        
        self.theta_inc = best_value[0] 
        #initialise these values based on the first set of configuration
        self.theta_inc_performance = best_value[1] 
        
  
            
    def fit_model(self) -> None:
        """
        Fits the internal surrogate model on the complete run list.
        """
        if len(self.R) > 0:
            
            #scaling of data
            configurations = [item[0] for item in self.R]
            X_df = pd.DataFrame(configurations)
        
            categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
            numerical_transformer = StandardScaler()
            categorical_cols, numerical_cols = identify_categorical_numerical(X_df)
            preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
            )
            
            #prepare for training
            X= X_df.iloc[::]
            
            
            y = np.array([performance for _, performance in self.R])  # Error rates
            
            # Define a Gaussian Process model
            kernel = C(1.0, (1e-4, 1e1)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e1), nu=2.5)

            # Initialize GaussianProcessRegressor with the Matern kernel
            internal_surrogate_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6)  # Alpha is noise level
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', internal_surrogate_model)
            ])
            
            self.internal_surrogate_model = pipeline.fit(X, y)
      
            # Fit the model to the observations
            
            return self.internal_surrogate_model

    def select_configuration(self) -> ConfigSpace.Configuration:
        """
        Determines which configurations are good, based on the internal surrogate model.
        Note that we are minimizing the error, but the expected improvement takes into account that.
        Therefore, we are maximizing expected improvement here.

        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        """
        
        
        # candidates = [self.config_space.sample_configuration().get_dictionary() for _ in range(20)]  
        
        # candidates = [clean_configuration(self.config_space, each_candidate) for each_candidate in candidates]
        
        # candidate_configs = pd.DataFrame([candidate for candidate in candidates])
        
        # theta = candidate_configs.iloc[::] 
        
        # Calculate the expected improvement for each candidate
        thetas = self.sample_space.sample(n=20)
        ei_values = self.expected_improvement(self.internal_surrogate_model, self.theta_inc_performance, thetas)
        
        # Select the configuration with the highest expected improvement
        best_candidate_idx = np.argmax(ei_values)
        best_candidate = thetas.iloc[best_candidate_idx]
      
        return best_candidate

    @staticmethod
    def expected_improvement(model_pipeline: Pipeline, f_star: float, theta: np.array) -> np.array:
        """
        Acquisition function that determines which configurations are good and which
        are not good.

        :param model_pipeline: The internal surrogate model (should be fitted already)
        :param f_star: The current incumbent (theta_inc)
        :param theta: A (n, m) array, each column represents a hyperparameter and each row
        represents a configuration
        :return: A size n vector, same size as each element representing the EI of a given
        configuration
        
        so, select configuration uses this function to get EI 
        """
        # Predict the mean and standard deviation for the candidates
        
        mu, sigma = model_pipeline.predict(theta, return_std=True)
        if f_star == None:
            f_star = mu
        xi = 0.03  # Exploration parameter
        f_star_exploration = f_star + xi
        sigma = np.maximum(sigma, 1e-9)  # Avoid division by zero
        
        sigma_noise = np.maximum(sigma, 1e-2)
        # Calculate the Expected Improvement (EI)
        # z = (f_star_exploration - mu) / sigma_noise
        z = (f_star - mu) / sigma
        
        # ei = (f_star_exploration - mu) * norm.cdf(z) + sigma_noise * norm.pdf(z)
        ei = (f_star- mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
        
    def optimize(self,  theta_new):
        #performance = test_real_world_model(self.config_space, theta_new)
        
        performance = self.ex_surrogate_class.external_surrogate_predict(theta_new)
        return performance
    
        
    def update_runs(self, run: typing.Tuple[typing.Dict, float]):
        """
        After a configuration has been selected and ran, it will be added to the run list
        (so that the model can be trained on it during the next iterations).

        :param run: A tuple (configuration, performance) where performance is error rate
        """
        
        #adding random candidate to R for new exploration area to avoid stagnant behaviour
        # random_candidate = self.config_space.sample_configuration().get_dictionary()  
        
        # candidate = clean_configuration(self.config_space, random_candidate)
        
        # random_cand_perf = smbo.optimize(candidate)
        
        # self.R.append((candidate, random_cand_perf))
        
        self.R.append(run)
        
        
        if run[1] < self.theta_inc_performance:
            self.theta_inc_performance = run[1]
            self.theta_inc = run[0]
        
        self.result_performance.append(self.theta_inc_performance)

import pickle

class ExternalSurrogate():
    def __init__(self, args, config_space):
        df_data  = pd.read_csv(args.configurations_performance_file)
        
        self.sg = SurrogateModel(config_space)
        
        self.sg.model =  self.sg.fit(df_data)
      
        
    def external_surrogate_predict(self, theta):
        theta_val = dict(theta)
        error_rate = self.sg.predict(theta_val)
        return error_rate
    
    
    
    def get_initial_sample_config(self, config_space):
            capital_phi = []
            
            
            for _ in range(20):  # Sample 20 initial configurations
                config = config_space.sample_configuration()    
                self.sg.config_space = config_space
                theta_val = dict(config)
                error_rate = self.sg.predict(theta_val)
               
                capital_phi.append((theta_val, error_rate))
                
            return capital_phi


def plot_value(total_budget, all_performances):
    
    plt.figure(figsize=(6, 6))
    plt.plot(range(total_budget), all_performances, color='blue', label='Iterative Best Performances so far')
    

    
    min_performance = min(all_performances)

    min_budget = all_performances.index(min_performance)

    plt.scatter(min_budget, min_performance, color='red', zorder=5, label=' Best Performance')

    # Add a horizontal line at the minimum performance point
    plt.axhline(y=min_performance, color='red', linestyle='--', label=f'Best Score: {min_performance:.6f}')
    # Get the current y-ticks
    yticks = list(plt.yticks()[0])

    # Add the minimum performance to the y-ticks if it's not already there
    if min_performance not in yticks:
        yticks.append(min_performance)
        yticks = sorted(yticks)  # Sort the ticks to keep them in order

    # Set the updated y-ticks
    plt.yticks(yticks)
    plt.xlabel('Budget')
    plt.ylabel('Guassian Regressor Performance')
    plt.title('Performances tracked during SMBO')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    

    
def identify_categorical_numerical( df):
        categorical_cols = []
        numerical_cols = []

        for col in df.columns:
            # If the column has an object data type or boolean (which could represent categories), treat as categorical
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                categorical_cols.append(col)
            # Otherwise, it's treated as numerical
            else:
                numerical_cols.append(col)
        
        return categorical_cols, numerical_cols
    
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv')
    
    return parser.parse_args()

def input_preprocessor(X_df):
    
    categorical_cols, numerical_cols = identify_categorical_numerical(X_df)
    
    # Define transformers
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def dynamic_preprocessor(X_df):
    """
    A wrapper for creating the preprocessor based on the data passed to the pipeline.
    """
    preprocessor = input_preprocessor(X_df)
    return preprocessor

def clean_configuration(config_space, theta):
    default_values = {}

    # Get the default value
    
    for key, _ in config_space.items():
        default_values[key] = config_space.get_hyperparameter(key).default_value
    
    
    # Find missing keys in the partial configuration
    missing_keys = [key for key in default_values.keys() if key not in theta]
    
    # Add missing keys with default values to the configuration
    for key in missing_keys:
        theta[key] = default_values[key]
        
        
    return theta

class SuccessiveHalvingBasedOptimization(object):

    def __init__(self, config_space, no_samples):
        """
        Initializes empty variables for the model, best performance and configuration , the first set of thetas and considering the total samples as 4000 for experiments
        """
        self.config_space = config_space
        self.best_configuration = None
        self.best_performance = None
        self.result_performance = []
        self.model = None
        
        self.thetas = None
        self.total_samples = no_samples
        self.all_performances = None
        self.anchors_travelled = None
        self.all_thetas = None
        
        
    def initialize(self) -> None:
        """
        Initializes the model by training on the entire set of datasets with all the anchor sizes. 
        
        """
        df_data  = pd.read_csv(args.configurations_performance_file)
        surrogate_model = SurrogateModel(self.config_space)
        
        self.model = surrogate_model.fit(df_data)
        print("Trained the model across all anchor sizes")
        anchor_sizes = df_data['anchor_size'].unique()
        self.anchor_sizes = sorted(anchor_sizes)
        
        
   
    def create_initial_configurations(self) -> ConfigSpace.Configuration:
        """

        :return: thetas: A size n vector, containing all configurations handling any missing columns from sample configs
        """
        
        #based on the sample space, finding the  iterations after which only one config will be remaining unless we are out of anchor sizes/resoruces.
        
        self.total_iterations = len(bin(self.total_samples)[2:])
        
        
        candidates = [dict(self.config_space.sample_configuration()) for _ in range(self.total_samples)]  
        
        #add default values in missing columns from sample configurations
        
        candidates = [clean_configuration(self.config_space, each_candidate) for each_candidate in candidates]
    
        candidate_configs = pd.DataFrame([candidate for candidate in candidates])
        
        thetas = candidate_configs.iloc[::]  
        return thetas


    def pruning(self, performances):
        """
        get the top half best performances and return the positions
        """
        sorted_indices = np.argsort(performances)[:len(performances)//2]
        return sorted_indices
        

    
    def get_anchors(self,full_training_size, total_iterations, threshold):
        
        """
        get the anchors for the fixed iterations based on no of configurations. experimenting till full anchor set size
        """
        n = full_training_size
        halved_values = [full_training_size]
        iteration = 1
        while iteration <=total_iterations and int(n//2) > threshold:
            
            n = int(n // 2)
            halved_values.append(n)
            iteration +=1
        return halved_values[::-1]
   
    def prune_update_optimize(self, thetas):
        """
        Finalize the anchor sizes to experiment considering the full training set size. 
        After evaluation of the configuration using the model, find the top half of the perofrmances with best scores (in our case minimum error rates).
        Select the batch for next iteration with an increased anchor size

        :param thetas: all the initial set of sample configurations 
        
        """
        
         
        #list of anchor sizes for the given iterations including the full training set size
        
        self.possible_anchors = self.get_anchors(max(self.anchor_sizes), self.total_iterations, min(self.anchor_sizes))
        
    
      
        iteration = 0

        current_performances_index = np.full(len(thetas), np.nan)
        thetas1 = thetas.copy()
        
        all_performances = []
        anchors_travelled = []
        
    
        
        for anchor_ in self.possible_anchors:
            
            thetas1['anchor_size'] = anchor_
            performances = self.model.predict(thetas1)
           
            if iteration == 0:
                current_performances_index = performances
               
            
            positions = self.pruning(performances)
            sorted_performances = performances[positions]
            
            thetas1 = thetas1.iloc[positions]
            
            current_index_list = thetas1.index.to_list()
            current_performances_index = np.full(len(thetas), np.nan)
            current_performances_index[current_index_list] = sorted_performances
            all_performances.append(current_performances_index)
            
            

            anchors_travelled.append(anchor_)
            if len(self.possible_anchors)-1 == iteration:
                
                self.best_configuration = thetas1.iloc[0]
                self.best_performance = sorted_performances[0]
                
                self.best_anchor = anchor_
                return all_performances,anchors_travelled
            
            iteration +=1
        
 
       
def eval_successive(args, no_samples):
    """
    evaluate successive halving -finidng best configuraiton
    """
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    config_space.seed(42)
    s_h_method =  SuccessiveHalvingBasedOptimization(config_space=config_space, no_samples=no_samples)

    
    s_h_method.initialize()
    thetas = s_h_method.create_initial_configurations()
    
    all_performances, anchors_travelled = s_h_method.prune_update_optimize(thetas)
    
   
    
    s_h_method.all_performances = all_performances
    s_h_method.all_thetas = thetas
    s_h_method.anchors_travelled = anchors_travelled
    
    return s_h_method, s_h_method.best_configuration, s_h_method.best_performance
   

def eval_smbo(args, max_anchor, total_budget,  s_h_method, best_sh_configuration, best_sh_performance):
    """
    evaluate smbo on 16000 anchor size and plot the best configuration for both smbo and successive halving -> their error and iteration number on which it was identified
    """
    sample_space = s_h_method.all_thetas

    
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    
    
    sample_space['anchor_size'] = max_anchor
    
    
 
    
    

    anchor_size =  Constant("anchor_size", max_anchor)
    
    config_space.add_hyperparameter(anchor_size)
    ex_surrogate_class = ExternalSurrogate(args=args, config_space=config_space)
    capital_phi = ex_surrogate_class.get_initial_sample_config(config_space)
    current_performances_index = np.full(len(sample_space), np.nan)
    smbo = SequentialModelBasedOptimization(config_space, max_anchor, ex_surrogate_class, sample_space, current_performances_index )
    
    smbo.initialize(capital_phi)
    
    budget_left = total_budget
    all_status = []
    all_indexes = []
    i = 4
    
    while budget_left:
        current_performances_index = np.full(len(sample_space), np.nan)
        smbo.fit_model()
        theta_new = smbo.select_configuration() 
        
        performance = smbo.optimize(theta_new)
        
        #if performance < smbo.theta_inc_performance:
        current_performances_index[theta_new.name] = performance
        
        
        smbo.update_runs((theta_new , performance))
        smbo.all_performances.append(performance)
        all_status.append(current_performances_index)
        all_indexes.append(theta_new.name)
        budget_left = budget_left-1
        try:
            best_index = smbo.theta_inc.name
        except:
            #if no theta identified, run the loop again, increasing the budget
            budget_left = budget_left + 1
            total_budget = total_budget + 1
    print(min(smbo.result_performance))
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame(all_status)
    df.columns = [f"Config_{i}" for i in range(len(all_status[0]))]
    
    
    
    df_sh = pd.DataFrame(s_h_method.all_performances)
    df_sh.columns = [f"Config_{i}" for i in range(len(s_h_method.all_performances[0]))]
    
    
    len_smbo = len(df[f'Config_{best_index}'])
    len_sh = len(df_sh[f'Config_{best_sh_configuration.name}'])

    # Find the maximum length to pad the shorter list
    max_len = max(len_smbo, len_sh)

    # Pad both lists to the same length with NaN values for the shorter list
    smbo_values = np.pad(df[f'Config_{best_index}'], (0, max_len - len_smbo), constant_values=np.nan)
    sh_values = np.pad(df_sh[f'Config_{best_sh_configuration.name}'], (0, max_len - len_sh), constant_values=np.nan)
    plt.plot(list(range(max_len)), smbo_values, marker='o', label=f'SMBO : Config_{best_index}')
    
    plt.plot(list(range(max_len)), sh_values, marker='*', label=f'SH:Config_{best_sh_configuration.name}')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))  # `nbins` controls how many ticks to display

    # Rotate x-axis labels if necessary
    plt.xticks(ticks= list(range(max_len)),labels= list(range(max_len)))  # Set iteration points on the x-axis
    
   
    plt.axhline(y=best_sh_performance, color='red', linestyle='--', label=f'Best Score SH : {best_sh_performance:.6f}')
    plt.axhline(y = min(smbo.result_performance), color='green', linestyle='--', label=f'Best Score SMBO : {min(smbo.result_performance):.6f}')
    
    yticks = list(plt.yticks()[0])
    
    yticks.append(best_sh_performance)
    yticks.append(min(smbo.result_performance))
    yticks = sorted(yticks)  # Sort the ticks to keep them in order
    plt.ylim(bottom=0, top = 0.1)
    
    plt.xlabel('Iterations')
    plt.ylabel('Performance')
    plt.title('Performance of Configurations over Iterations')

    # Add a legend to identify each configuration
    plt.legend(title="Configurations", loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title_fontsize='small')


    # Display the plot
    plt.grid(True)
    plt.tight_layout()
    
    plt.show()

 
if __name__ == '__main__':
    #no of samples = 4000 with maximum anchor of 16000 and total budget for smbo is 30
    args = parse_args()
    no_samples = 4000
    
    s_h_method, best_sh_configuration, best_sh_performance = eval_successive(args, no_samples)
    
    
    max_anchor = 16000
    total_budget = 30
    result_performances = eval_smbo(args, max_anchor, total_budget,  s_h_method, best_sh_configuration, best_sh_performance)
    
    