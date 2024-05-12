from enum import Enum

def form_strategy_0(min_i, min_j, max_i, max_j):
    """
          *
        
        
       *     * 
    """
    h_upper = min_i + (max_i - min_i) // 5
    h_lower = min_i + (max_i - min_i) * 0.8
    w_half = (min_j + max_j) // 2
    w_left = min_j + (max_j - min_j)//4
    w_right = min_j + (max_j - min_j)* 0.75

    return [
        [h_upper, w_half],
        [h_lower, w_left],
        [h_lower, w_right]
    ]

def form_strategy_1(min_i, min_j, max_i, max_j):
    """
        -*-    -*-
        
        
        -*-    -*-
    """
    h_upper = min_i + (max_i - min_i) // 5
    h_lower = min_i + (max_i - min_i) * 0.8
    w_half = (min_j + max_j) // 2
    w_left = min_j + (max_j - min_j)//4
    w_right = min_j + (max_j - min_j)* 0.75

    return [
        [h_upper, w_left],
        [h_upper, w_right],
        [h_lower, w_left],
        [h_lower, w_right],
    ]

def form_strategy_2(min_i, min_j, max_i, max_j):
    """
        --*--  
        
        
        --*-- 
    """
    h_upper = min_i + (max_i - min_i) // 5
    h_lower = min_i + (max_i - min_i) * 0.8
    w_half = (min_j + max_j) // 2
    w_left = min_j + (max_j - min_j)//4
    w_right = min_j + (max_j - min_j)* 0.75
    
    return [
        [h_upper, w_half],
        [h_lower, w_half],
    ]

def form_strategy_3(min_i, min_j, max_i, max_j):
    """
         |
        -*-
         |
    """
    h_upper = min_i + (max_i - min_i) // 5
    h_lower = min_i + (max_i - min_i) * 0.8
    h_half = (min_i + max_i) // 2
    w_half = (min_j + max_j) // 2
    w_left = min_j + (max_j - min_j)//4
    w_right = min_j + (max_j - min_j)* 0.75
    
    return [
        [h_half, w_half],
    ]

class PointsFormingSamStrategies(Enum):
    strategy_0 = form_strategy_0
    strategy_1 = form_strategy_1
    strategy_2 = form_strategy_2
    strategy_3 = form_strategy_3
