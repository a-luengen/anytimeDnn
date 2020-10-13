from resnet.utils import getRandomBoolListPermutation

SKIP_POLICY = {'policy': None}

class DenseNetDropPolicy(object):
    def __init__(self, block_config):
        self.block_config = block_config
        self.block_layer_config = []
    
    def getDropLayerConfiguration(self, layer_id: int):
        raise NotImplementedError("Function should be called from within its child classes.")
        #return self.block_layer_config[layer_id]
    
    def getFullConfig(self):
        return self.block_layer_config


class DNDropRandNPolicy(DenseNetDropPolicy):
    def __init__(self, block_config, n):
        super(DNDropRandNPolicy, self).__init__(block_config)

        max_layers = sum(block_config)

        if n > max_layers:
            raise ValueError('Value for n is to heigh. Cannot drop more Layers than possible.')

        self._n = n   

        # generate random 
        temp_perm = getRandomBoolListPermutation(max_layers, n)

        prev_val = 0
        temp_split = []
        for i in block_config:
            temp_split.append((prev_val, prev_val + i))
            prev_val += i
        
        self.block_layer_config = [temp_perm[i:j] for i, j in temp_split]
    
    def getDropLayerConfiguration(self, layer_id: int):
        return self.block_layer_config[layer_id]

def setSkipPolicy(policy):
    SKIP_POLICY['policy'] = policy

def getSkipPolicy() -> DenseNetDropPolicy:
    return SKIP_POLICY['policy']