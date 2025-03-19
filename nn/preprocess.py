# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #determine class imbalance

    #separate positive and negative samples
    pos_seqs = [seq for seq, label in zip(seqs, labels) if label] # list comprehension
    neg_seqs = [seq for seq, label in zip(seqs, labels) if not label]

    #count positive and negative samples
    num_pos = len(pos_seqs)
    num_neg = len(neg_seqs)

    #handle empty class
    if num_pos == 0 or num_neg == 0:
        return [],[]

    #resample smaller class to match larger class
    num_samples = max(num_pos, num_neg)
    
    # if more positive samples than negative samples
    if num_pos > num_neg:
        #sample negative samples with replacement
        neg_seqs = list(np.random.choice(neg_seqs, size=num_samples, replace=True))
        
    else: #more negative samples than positive samples
        # sample positive samples with replacement
        pos_seqs = list(np.random.choice(pos_seqs, size=num_samples, replace=True))
    
    
    #combine positive and negative samples
    sampled_seqs = pos_seqs + neg_seqs
    num_samples = len(sampled_seqs)
    sampled_labels = [True] * len(pos_seqs) + [False] * len(neg_seqs)

    #shuffle the samples - ensure random order
    combined = list(zip(sampled_seqs, sampled_labels))
    np.random.shuffle(combined)

    sampled_seqs, sampled_labels = zip(*combined)

    return list(sampled_seqs), list(sampled_labels)

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # define dictionary for one-hot encoding
    encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

    # encode sequences
    encodings = []

    '''
    
    for seq in seq_arr:
        encodings.append([encoding_dict[base] for base in seq])
        return np.array(encodings).flatten()
  
    '''
    for seq in seq_arr:
        encoded_seq = np.concatenate([encoding_dict[base] for base in seq])  # flatten the encoding
        encodings.append(encoded_seq)
    
    return np.array(encodings)  # shape is num_samples, seq_length * 4) 

    
   
    
