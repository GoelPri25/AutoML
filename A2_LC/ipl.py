import logging
import numpy as np
import typing
import ConfigSpace
from ConfigSpace import Constant
from scipy.optimize import curve_fit

from vertical_model_evaluator import VerticalModelEvaluator

class IPL(VerticalModelEvaluator):
    
    def __init__(self, surrogate_model, minimal_anchor, final_anchor, anchor_sizes):
        """
        Inherit from VerticalModelEvaluator and initialize parameters.
        """
        super().__init__(surrogate_model, minimal_anchor, final_anchor, anchor_sizes)
    
    def _inverse_power_law(self, s, alpha, beta, gamma):
        """
        The inverse power law function: μ_s = alpha + beta * s^(-gamma)
        
        :param s: anchor size
        :param alpha: Baseline constant.
        :param beta: Scaling parameter.
        :param gamma: Power-law exponent.
        :return: Predicted value.
        """
        return alpha + beta * np.power(s, -gamma)
    
    
    def ipl_fit(
        self, evaluations
    ):
        
        fixed_schedules, performances = zip(*evaluations)

        # Convert zip object to lists
        fixed_schedules = list(fixed_schedules)
        performances = list(performances)


        X = np.array(fixed_schedules)
        y = np.array(performances)
        params, _ = curve_fit(self._inverse_power_law, X, y, maxfev=30000)
        
        return params
    
    def ipl_extrapolation(self, params, target_anchor):
        alpha,beta,gamma = params
        extrapolated_target_performance  = self._inverse_power_law(target_anchor, alpha, beta, gamma)
        return extrapolated_target_performance
        
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
        print("here", best_so_far)
        """
        
        Does a staged evaluation of the model, on increasing anchor sizes.
        Determines after the evaluation at every anchor an optimistic
        extrapolation. In case the optimistic extrapolation can not improve
        over the best so far, it stops the evaluation.
        In case the best so far is not determined (None), it evaluates
        immediately on the final anchor (determined by self.final_anchor)

        :param best_so_far: indicates which performance has been obtained so far
        :param configuration: A dictionary indicating the configuration

        :return: A tuple of the evaluations that have been done. Each element of
        the tuple consists of two elements: the anchor size and the estimated
        performance.
        """
        
     

        evaluations = []
        if best_so_far is None:
            configuration['anchor_size'] = self.final_anchor
            final_performance = self.surrogate_model.predict(configuration)
            
            best_so_far = final_performance
            print("At none ", best_so_far)     
            evaluations.append((self.final_anchor, best_so_far))
        for current_anchor in self.anchor_sizes:
            configuration['anchor_size'] = current_anchor
            current_performance = self.surrogate_model.predict(configuration)
            print(current_performance)
            evaluations.append((current_anchor, current_performance))
            
            
        params = self.ipl_fit(evaluations)
        print(params)
        extrapolated_performance = self.ipl_extrapolation(params, self.final_anchor)
        
        #if extrapolated_performance > best_so_far:  #since in our case, lower is better
            #print(f"Best so far is {best_so_far} , couldnt find more, {current_performance}, {current_anchor}")
            #print(current_anchor)
            
               
            #print("Best so far", best_so_far)
            #update anchors
        print("Best so far ", best_so_far)
        print("extrapolated value", extrapolated_performance, current_performance)
        if extrapolated_performance <= best_so_far:
            configuration['anchor_size'] = self.final_anchor
            final_performance = self.surrogate_model.predict(configuration)
            evaluations.append((self.final_anchor, final_performance))
            #print(final_performance, best_so_far)
            
            #print(sorted(evaluations, key= lambda x: x[1], reverse=True))
            
        evaluations = sorted(evaluations, key= lambda x: x[1], reverse=True)
        
        #print(evaluations)
        #print(evaluations)
        return evaluations
        #raise NotImplemented()
