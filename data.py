"""
author-gh: @adithya8

editor-gh: ykl7
"""

import collections

import numpy as np
from scipy.misc import central_diff_weights
import torch

np.random.seed(1234)
torch.manual_seed(1234)

# Read the data into a list of strings.
def read_data(filename):
    with open(filename) as file:
        text = file.read()
        data = [token.lower() for token in text.strip().split(" ")]
    return data

def build_dataset(words, vocab_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # token_to_id dictionary, id_to_taken reverse_dictionary
    vocab_token_to_id = dict()
    for word, _ in count:
        vocab_token_to_id[word] = len(vocab_token_to_id)
    data = list()
    unk_count = 0
    for word in words:
        if word in vocab_token_to_id:
            index = vocab_token_to_id[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    vocab_id_to_token = dict(zip(vocab_token_to_id.values(), vocab_token_to_id.keys()))
    return data, count, vocab_token_to_id, vocab_id_to_token

class Dataset:
    def __init__(self, data, batch_size=128, num_skips=8, skip_window=4):
        """
        @data_index: the index of a word. You can access a word using data[data_index]
        @batch_size: the number of instances in one batch
        @num_skips: the number of samples you want to draw in a window 
                (In the below example, it was 2)
        @skip_windows: decides how many words to consider left and right from a context word. 
                    (So, skip_windows*2+1 = window_size)
        """

        self.data_index=0
        self.data = data
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
    
    def reset_index(self, idx=0):
        self.data_index=idx

    def generate_batch(self):
        """
        Write the code generate a training batch

        batch will contain word ids for context words. Dimension is [batch_size].
        labels will contain word ids for predicting(target) words. Dimension is [batch_size, 1].
        """
        
        center_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        context_word = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        
        # stride: for the rolling window
        stride = 1 

        ### TODO(students): start
        # window_size = self.skip_window*2 + 1
        batch_size_counter = 0
        center_index = 0
        context_index = 0
        
        d = self.data_index
        for d in range(len(self.data)):
            windowLeft = []
            windowRight = []
            if(d > self.skip_window):
                windowLeft = self.data[d-self.skip_window:d]
                if((d+self.skip_window) > (len(self.data)-1)):
                    i = d+1
                    while i < (len(self.data)):
                        windowRight.append(self.data[i])
                        i+=1
                else:
                    windowRight = self.data[d+1:d+self.skip_window+1]
            else:
                i = 0
                while i != d:
                    windowLeft.append(self.data[i])
                    i+=1
                windowRight = self.data[d+1:d+self.skip_window+1]

            for element in range(len(windowLeft)):
                center_word[center_index] = self.data[d]
                context_word[context_index] = windowLeft[element]
                center_index+=1
                context_index+=1
                batch_size_counter+=1
                if(batch_size_counter == self.batch_size):
                    break

            if(batch_size_counter != self.batch_size):
                for element in range(len(windowRight)):
                    center_word[center_index] = self.data[d]
                    context_word[context_index] = windowRight[element]
                    center_index+=1
                    context_index+=1
                    batch_size_counter+=1
                    if(batch_size_counter == self.batch_size):
                        break
            else:
                break

        self.data_index = d
        ### TODO(students): end

        return torch.LongTensor(center_word), torch.LongTensor(context_word)