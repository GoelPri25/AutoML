import argparse
import ConfigSpace
import logging
import matplotlib.pyplot as plt
import pandas as pd
from surrogate_model import SurrogateModel
from ipl import IPL
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_space_file', type=str, default='lcdb_config_space_knn.json')
    parser.add_argument('--configurations_performance_file', type=str, default='config_performances_dataset-1457.csv')
    # max_anchor_size: connected to the configurations_performance_file. The max value upon which anchors are sampled
    parser.add_argument('--minimal_anchor', type=int, default=256)
    parser.add_argument('--max_anchor_size', type=int, default=16000)
    parser.add_argument('--num_iterations', type=int, default=50)

    return parser.parse_args()


def run(args):
    config_space = ConfigSpace.ConfigurationSpace.from_json(args.config_space_file)
    df = pd.read_csv(args.configurations_performance_file)
    max_anchor_size = max(df['anchor_size'].unique())
    min_anchor_size = min(df['anchor_size'].unique())
    surrogate_model = SurrogateModel(config_space)
    surrogate_model.fit(df)
    ####
    print(sorted(df['anchor_size'].unique()))
    fixed_schedules = sorted(df['anchor_size'].unique())
    fixed_schedules = [16, 32, 64, 128, 256, 362]
    print(fixed_schedules)
    ####
  
    ipl = IPL(surrogate_model, min_anchor_size, max_anchor_size, fixed_schedules)
    best_so_far = None
    
    
    for idx in range(args.num_iterations):
        theta_new = dict(config_space.sample_configuration())
        #print(theta_new)
        
        result = ipl.evaluate_model(best_so_far, theta_new)
        final_result = result[-1][1]
        
        if best_so_far is None or final_result < best_so_far:
            best_so_far = final_result
            print("Best so far ", best_so_far)
        x_values = [i[0] for i in result]
        y_values = [i[1] for i in result]
        plt.plot(x_values, y_values, "-o")


    plt.xlabel('Anchor size')
    plt.ylabel('Score')
    plt.title('IPL-based extrapolation on LCDB dataset \n Max anchor size: ' + str(args.max_anchor_size))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    run(parse_args())
