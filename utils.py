def argmax(random_generator, q_values):
    '''
    A function that chooses the greatest value in a sequence
    and breaks ties at random
    '''
    top = float("-inf")
    ties = []
    
    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []
        if q_values[i] == top:
            ties += [i]
        
    index = random_generator.choice(ties)
    return index
