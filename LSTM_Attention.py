#!/usr/bin/env python3
"""
student.py

Group    : very big brains (g023473)
Members  : Ming Xuan CHUA z5159352, Qie Shang PUA z5157686
Course   : UNSW COMP9444 Neural Networks and Deep Learning



"""

import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as toptim
from torchtext.vocab import GloVe

import re
import nltk
import numpy as np
nltk.download('stopwords') # Download stopwords from NLTK  
from nltk.corpus import stopwords
from config import device


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    # https://piazza.com/class/kf3o1qjasxgxn?cid=237
    sample = re.sub('[!?.,@#$%\^*();:/~<>]', '', sample)
    sample = sample.replace('-', ' ')
    sample = sample.replace("&", 'and')
    processed = sample.split()

#     lemmatizer=WordNetLemmatizer()
#     for word in processed:
#         word = lemmatizer.lemmatize(word)
        
    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.

    """
    
#     print(sample)
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    
    return batch

stopWords = stopwords.words('english') # from NLTK
word_len = 300
wordVectors = GloVe(name='6B', dim=word_len)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    _, categoryOutput = torch.max(categoryOutput.data, 1)
    
    #round to 0 or 1
    ratingOutput = torch.round(ratingOutput)
    # return as long tensor
    return ratingOutput.long(), categoryOutput.long()

################################################################################
###################### The following determines the model ######################
################################################################################
class Attention(tnn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = tnn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            tnn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]
            
        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_size, 1)
                            )
    
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        
        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions

lstm_hidden_size = 128
lstm_layers = 1

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()

        # Don't need this layer because embedding is done by GloVe
        # self.embedding = nn.Embedding(len(text_field.vocab), 300)
        
        self.lstm = tnn.LSTM(input_size=word_len,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = tnn.Dropout(p=0.5)
        self.attention = Attention(lstm_hidden_size*2, batch_first=True)

        self.rating_fc = tnn.Linear(2*lstm_hidden_size, 1)
        self.category_fc = tnn.Linear(2*lstm_hidden_size, 5)

    def forward(self, input, length):
        
        # https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        # Dynamic Padding - not sure if needed?
        packed_input = pack_padded_sequence(input, length, batch_first=True, enforce_sorted=False)
        
        x, _ = self.lstm(packed_input)
        
        x, lengths = pad_packed_sequence(x, batch_first=True)   
        x, _ = self.attention(x, lengths) # skip connect
        # [30, 256]
        '''
        These also produce outputs of variable length, 
        but if you want to feed information into linear or other fixed size layers 
        then the last output / hidden state of an RNN can be used, 
        e.g. using a tensor slice to select the last element of a sequence.
        '''        
#         out_forward = x[range(len(x)), length - 1, :lstm_hidden_size]
#         out_reverse = x[:, 0, lstm_hidden_size:]
#         out_reduced = torch.cat((out_forward, out_reverse), 1)    
        
#         x = self.drop(out_reduced)
        
        # Rating
        rating_out = self.rating_fc(x)
        rating_out = torch.squeeze(rating_out) # remove dim with 1  
        rating_out = torch.sigmoid(rating_out) 
        
        # Category
        cat_out = self.category_fc(x)
        # Adding one more FC layer does not improve the category accuracy
        cat_out = F.relu(cat_out)
        
        return rating_out, cat_out
        
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/7
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        
        # Do not round to 0 or 1 here, only convert in the convertNetOutput method for Accuracy
        rating_loss = F.binary_cross_entropy(ratingOutput.float(), ratingTarget.float())
        cat_loss = F.cross_entropy(categoryOutput, categoryTarget)
        
        total_loss = rating_loss + cat_loss
        
        return total_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 5
optimiser = toptim.Adam(net.parameters(), lr=0.001)


     