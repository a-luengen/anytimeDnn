from resnet.utils import getRandomBoolListPermutation, getGaussDistributedBoolList

SKIP_POLICY = {'policy': None}

class DenseNetDropPolicy(object):
    def __init__(self, block_config):
        self.block_config = block_config
        self.blockSkippingConfiguration = []
    
    def getSkipConfigurationListOfBlock(self, layerIdx: int):
        raise NotImplementedError("Function should be called from within its child classes.")
    
    def getFullConfig(self):
        return self.blockSkippingConfiguration

class DenseNetDropNPolicy(DenseNetDropPolicy):
    def __init__(self, block_config, n: int):
        super(DenseNetDropNPolicy, self).__init__(block_config)
        
        max_layers = sum(block_config)

        if n > max_layers:
            raise ValueError('Value for n is to heigh. Cannot drop more Layers than possible.')
        self._n = n
        self.blockSkippingConfiguration = []
    
    def getSkipConfigurationListOfBlock(self, layerIdx: int):
        return self.blockSkippingConfiguration[layerIdx]

class DenseNetDropRandNPolicy(DenseNetDropNPolicy):

    name = 'skip'

    def __init__(self, block_config, n):
        super(DenseNetDropRandNPolicy, self).__init__(block_config, n)

        # generate random 
        temp_perm = getRandomBoolListPermutation(sum(block_config), n)

        prev_val = 0
        temp_split = []
        for i in block_config:
            temp_split.append((prev_val, prev_val + i))
            prev_val += i
        
        self.blockSkippingConfiguration = [temp_perm[i:j] for i, j in temp_split]

class DenseNetDropLastNPolicy(DenseNetDropNPolicy):
    """
        Drops a random amount of last layers within blocks, but always 
        a global amount of exactly n layers.
    """

    name = 'skip-last'

    def __init__(self, block_config, n:int):
        super(DenseNetDropLastNPolicy, self).__init__(block_config, n)

        temp_perm = getRandomBoolListPermutation(sum(block_config), n)
        
        prev_val = 0
        temp_split = []
        for i in block_config:
            temp_split.append((prev_val, prev_val + i))
            prev_val += i
        
        self.blockSkippingConfiguration = [temp_perm[i:j].tolist() for i, j in temp_split]
        self.blockSkippingConfiguration = [sorted(x) for x in self.blockSkippingConfiguration]
        
class DenseNetDropLastNBlockwisePolicy(DenseNetDropNPolicy):
    """
        Drops a total of N-Layers from the Network, by evenly dropping the last layer
        from each block, until N-Layers are dropped. Starting from the last block on.
    """

    name = 'skip-last-n-block'

    def __init__(self, block_config, n: int):
        super(DenseNetDropLastNBlockwisePolicy, self).__init__(block_config, n)

        layer_config = []
        for block in block_config:
            layer_config.append([False] * block)
        
        skips_to_take = n
        next_layer_to_pick_from = len(block_config) - 1

        while skips_to_take > 0:
            has_picked = False
            # pick layer to delete
            layer = layer_config[next_layer_to_pick_from]

            for i in reversed(range(len(layer))):
                if layer[i] == False:
                    layer[i] = True
                    has_picked = True
                    break

            next_layer_to_pick_from = (next_layer_to_pick_from - 1) % len(block_config) 
            if has_picked:
                skips_to_take -= 1

        self.blockSkippingConfiguration = layer_config

class DenseNetDropNormalDistributedNPolicy(DenseNetDropNPolicy):
    """
        Drops N-Layers from a DenseNet choosen by Normal-Distribution.
    """

    name = 'skip-norm-n'

    def __init__(self, block_config, n):
        super(DenseNetDropNormalDistributedNPolicy, self).__init__(block_config, n)


        layer_config = getGaussDistributedBoolList(sum(block_config), n)

        prev_val = 0
        temp_split = []
        for i in block_config:
            temp_split.append((prev_val, prev_val + i))
            prev_val += i
        
        self.blockSkippingConfiguration = [layer_config[i:j] for i, j in temp_split]


def setSkipPolicy(policy):
    SKIP_POLICY['policy'] = policy

def getSkipPolicy() -> DenseNetDropPolicy:
    return SKIP_POLICY['policy']
