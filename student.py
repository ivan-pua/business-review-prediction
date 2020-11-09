#!/usr/bin/env python3
"""
student.py

Group: very big brains (g023473)
1. Ming Xuan CHUA z5159352
2. Qie Shang PUA z5157686

UNSW COMP9444 Neural Networks and Deep Learning


# Lemmatization did not improve accuracy 
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize


"""

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe

import re
import nltk
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
    # https://www.geeksforgeeks.org/python-removing-unwanted-characters-from-string/
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

        self.rating_fc = tnn.Linear(2*lstm_hidden_size, 1)
        self.category_fc = tnn.Linear(2*lstm_hidden_size, 5)

    def forward(self, input, length):
        
        # https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
        # Dynamic Padding from Website
        # packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(input)
       
        '''
        These also produce outputs of variable length, 
        but if you want to feed information into linear or other fixed size layers 
        then the last output / hidden state of an RNN can be used, 
        e.g. using a tensor slice to select the last element of a sequence.
        '''
        out_forward = x[range(len(x)), length - 1, :lstm_hidden_size]
        out_reverse = x[:, 0, lstm_hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        
        x = self.drop(out_reduced)

        # Rating
        rating_out = self.rating_fc(x)
        rating_out = torch.squeeze(rating_out) # remove dim with 1  
        rating_out = torch.sigmoid(rating_out) 
        
        # Category
        cat_out = self.category_fc(out_reduced)
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
        
        # Do not round to 0 or 1 here, only convert in the convertNetOutput for Accuracy
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
epochs = 8
optimiser = toptim.Adam(net.parameters(), lr=0.001)


     