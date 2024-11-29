import logging
import numpy as np
import typing
import ConfigSpace
from ConfigSpace import Constant

from vertical_model_evaluator import VerticalModelEvaluator

class LCCV(VerticalModelEvaluator):
    
    @staticmethod
    def optimistic_extrapolation(
        previous_anchor: int, previous_performance: float, 
        current_anchor: int, current_performance: float, target_anchor: int
    ) -> float:
        """
        Does the optimistic performance. Since we are working with a simplified
        surrogate model, we can not measure the infimum and supremum of the
        distribution. Just calculate the slope between the points, and
        extrapolate this.

        :param previous_anchor: See name
        :param previous_performance: Performance at previous anchor
        :param current_anchor: See name
        :param current_performance: Performance at current anchor
        :param target_anchor: the anchor at which we want to have the
        optimistic extrapolation
        :return: The optimistic extrapolation of the performance
        """
        slope = float((current_performance - previous_performance)/(current_anchor - previous_anchor))
        extrapolated_performance = current_performance + slope * (target_anchor - current_anchor)
        return extrapolated_performance
        #raise NotImplemented()
    
    def evaluate_model(self, best_so_far: typing.Optional[float], configuration: typing.Dict) -> typing.List[typing.Tuple[int, float]]:
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
        previous_performance  = None
        previous_anchor = None
        evaluations = []
        if best_so_far == None:
            configuration['anchor_size'] = self.final_anchor
            final_performance = self.surrogate_model.predict(configuration)
            
            best_so_far = final_performance
            #print("initial best ", best_so_far)
            #evaluations =  [(self.final_anchor, final_performance)]
            
        
        for current_anchor in self.anchor_sizes:
            configuration['anchor_size'] = current_anchor
            current_performance = self.surrogate_model.predict(configuration)
            evaluations.append((current_anchor, current_performance))
            if previous_performance is None:
                previous_performance = current_performance
                previous_anchor = current_anchor
                continue
            
            extrapolated_performance = self.optimistic_extrapolation(previous_anchor, previous_performance, current_anchor, current_performance, self.final_anchor)
            #print(f"current performance: {current_performance} and extrapolated performance: {extrapolated_performance}")
            if extrapolated_performance >= best_so_far:  #since in our case, lower is better
                print(f"Best so far is {best_so_far} , couldnt find more, {current_performance}, {current_anchor}")
                #print(current_anchor)
                
                break    
            #print("Best so far", best_so_far)
            #update anchors
            previous_anchor = current_anchor
            previous_performance = current_performance
        evaluations = sorted(evaluations, key= lambda x: x[1], reverse=True)
        #print(evaluations)
        return evaluations
        #raise NotImplemented()
