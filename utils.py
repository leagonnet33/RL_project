import numpy as np

def argmax(q_values):
    top = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        
        if q_values[i] > top:
            top = q_values[i]
            ties = []
        if q_values[i] == top:
            ties += [i]
        
    index = np.random.choice(ties)
    
    return index
