import argparse
import smbo
import example_run_experiment as RS
import matplotlib.pyplot as plt
import numpy as np
import successive_halving


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--model_path', type=str, default='external_surrogate_model.pkl')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-6.csv') #using the dataset with max anchor 16000
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=200)

    return parser.parse_args()


class Aggregator():
    '''
    An aggregator class that pulls together the results
    of Random Search and SMBO for comparison plots.
    '''

    def __init__(self):
        self.args = parse_args()
        self.smbo_result_mean = []
        self.RS_result_mean = []
        self.SH_result_mean = []

    def call_smbo(self, idx):
        for x in range(0,idx):
            smbo_result = smbo.eval_smbo(self.args, max_anchor=self.args.max_anchor_size, total_budget=self.args.num_iterations).result_performance
            self.smbo_result_mean.append(np.array(smbo_result))

    def call_RS(self, idx):
        for x in range(0,idx):
            RS_result = RS.run(self.args, aggregator_request=True, test_dataset=self.args.configurations_performance_file)
            self.RS_result_mean.append(np.array(RS_result['random_search']))

    
    def get_means_and_scores(self):
        '''
        Compute the mean values across 4 runs.
        '''
        #get a list of mean values over n iterations for each algorithm 
        mean_values_RS = [np.mean(scores) for scores in zip(*self.RS_result_mean)]
        mean_values_smbo = [np.mean(scores) for scores in zip(*self.smbo_result_mean)]

        #get variance for each mean 
        std_RS = [np.std(scores) for scores in zip(*self.RS_result_mean)]
        std_smbo = [np.std(scores) for scores in zip(*self.smbo_result_mean)]

        #get the best score and iteration index for each algorithm
        best_RS = min(mean_values_RS)
        best_smbo = min(mean_values_smbo)
        iteration_RS = mean_values_RS.index(best_RS)
        iteration_smbo = mean_values_smbo.index(best_smbo)

        #return mean values, variance, best scores, iteration index
        return mean_values_RS, mean_values_smbo, std_RS, std_smbo, best_RS, best_smbo, iteration_RS, iteration_smbo

    def plot_smbo_RS(self):
        '''
        Places the mean values of SMBO and Random Search in one plot.
        '''
        mean_values_RS, mean_values_smbo, std_RS, std_smbo, best_RS, best_smbo, iteration_RS, iteration_smbo = self.get_means_and_scores()

        plt.plot(range(len(mean_values_RS)-1), mean_values_RS[1:], label='Random Search mean scores')
        plt.scatter(iteration_RS, best_RS, color='red', label='Mean best score Random search=' + str(round(best_RS,3))) #zorder=5, label=' Best score='+str(round(smallest_score,3))+", \n iteration="+str(earliest_index)) 
        plt.fill_between(range(len(mean_values_RS)-1), (np.array(mean_values_RS[1:]) + np.array(std_RS[1:])), (np.array(mean_values_RS[1:]) - np.array(std_RS[1:])), alpha=0.3)

        plt.plot(range(len(mean_values_smbo)), mean_values_smbo, label='SMBO mean scores')
        plt.scatter(iteration_smbo, best_smbo, marker="^", color='red', label='Mean best score SMBO=' + str(round(best_smbo,3)))     
        plt.fill_between(range(len(mean_values_smbo)), (np.array(mean_values_smbo) + np.array(std_smbo)), (np.array(mean_values_smbo) - np.array(std_smbo)), alpha=0.3)
   
        plt.xlabel('Num iterations')
        plt.ylabel('Score')
        plt.title('Random Search and SMBO performance on the Letter dataset \n Max anchor size: ' + str(self.args.max_anchor_size))
        
        plt.legend()
        plt.tight_layout()
        plt.show()

        
   
   
if __name__=='__main__':
    agg = Aggregator()
    agg.call_smbo(4)
    agg.call_RS(4)
    #for smbo vs sh , please run smbo_vs_sh.py
    agg.plot_smbo_RS()
    
