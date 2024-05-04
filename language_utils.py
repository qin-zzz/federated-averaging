import numpy as np
import json

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

def get_shakespeare(json_path, num_client=100):
    inputs_lst = []
    targets_lst = []
    clients_lst = []
    data = {}

    with open(json_path, 'r') as inf:
        cdata = json.load(inf)
    data.update(cdata['user_data'])
    list_keys = list(data.keys())

    for (i, key) in enumerate(list_keys[:num_client]):
        # note: each time we append a list
        inputs = data[key]["x"]
        targets = data[key]["y"]

        for input_ in inputs:
            input_ = process_x(input_)
            inputs_lst.append(input_.reshape(-1))

        for target in targets:
            target = process_y(target)
            targets_lst += target[0]

        for _ in range(0, len(inputs)):
            clients_lst.append(i)

    return inputs_lst, targets_lst, clients_lst
      
def word_to_indices(word):
    '''returns a list of character indices

    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(max(0, ALL_LETTERS.find(c))) # added max to account for -1
    return indices

def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch

def process_y(raw_y_batch):
    y_batch = [word_to_indices(c) for c in raw_y_batch] 
    return y_batch