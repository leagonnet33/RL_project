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

def generate_rgb_color(random_generator):
    '''
    A function that generates RGB alpha values to be used as
    color a color kwarg in a matplotlib function.
    '''
    values = random_generator.random(size=4)
    return (values[0], values[1], values[2], .5 + values[3]/2) # Alpha should not be too small
    