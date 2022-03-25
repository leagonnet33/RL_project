import torch.nn as nn

class DenseModel(nn.Module):
    '''
    A series of linear layers
    '''
    def __init__(self, input_dimension, output_dimension) -> None:
        '''
        Initialize object
        '''
        super().__init__()
        # Build linear layers
        self.l1_ = nn.Linear(input_dimension, 10)
        self.l2_ = nn.Linear(10, 10)
        self.l3_ = nn.Linear(10, output_dimension)
        
        # Initialize weights
        nn.init.normal_(self.l1_.weight, std=.001)
        nn.init.normal_(self.l2_.weight, std=.001)
        nn.init.normal_(self.l3_.weight, std=.001)
    
    def forward(self, X):
        '''
        Forward pass of the state in the model
        '''
        output = nn.functional.relu(self.l1_(X))
        output = nn.functional.relu(self.l2_(output))
        return self.l3_(output)
        