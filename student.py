#!/usr/bin/env python3
"""
student.py

Group    : very big brains (g023473)
Members  : Ming Xuan CHUA z5159352, Qie Shang PUA z5157686
Course   : UNSW COMP9444 Neural Networks and Deep Learning

Before developing the model, the review text is preprocessed by removing
punctuations and replacing specific symbols. Stopwords are also added by
referring to the NLTK corpus. The review text is tokenized and embedded
using GloVe with a vector dimension of 300. For this model, a Bidirectional LSTM
is chosen because they are capable of learning long-term dependencies,
hence suitable for analysing a long review text. The vector embeddings
are passed through the Bi-LSTM layer. Then, the last hidden state of the
Bi-LSTM layer is passed into two different fully connected (linear) layers,
followed by their respective activation functions, to generate two output
tensors - rating and category. To convert the network’s output to the
predicted labels, rating output is rounded to 0 or 1, whereas the index of the
highest value in the category tensor is selected.  For the loss function,
binary cross entropy is assigned for rating whereas cross entropy is chosen
for category. Both loss functions are ideal candidates for classification tasks. 

Initially, a LSTM model is developed as a base model. Then, several design
improvements are incorporated into the base model and have proven to work,
including:
-   Using a bidirectional LSTM instead of uni-directional to obtain better
    context of the review text
-   Replacing SGD with Adam Optimiser, with a learning rate of 0.001
-   Increasing word vector size from 50 to 300 to represent the semantics better
-   Adding Dropout layer to prevent overfitting 
-   Preprocessing Data

However, some methods were unable to boost the accuracy, such as:
-   Attention Layer; it increased the score by 1% but consumed a lot of
    computational time and resources
-   Having more than 1 Bi- LSTM layer
-   Adding more Linear layers
-   Word Lemmatization 

Overall, this model is successful as it consistently achieves a weighted score
of 85% within 5 epochs.


"""

# Import packages
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as toptim
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import re
import numpy as np
from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    sample = re.sub('[!?.,@#$%\^*();:/~<>]', '', sample) # remove punctuations
    sample = sample.replace('-', ' ') # replace 
    sample = sample.replace("&", 'and') # replace
    processed = sample.split() # tokenise
        
    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.

    """
    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    
    return batch


# Stopwords are obtained from the latest NLTK corpus
stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

# Longer embedding size represents the semantics better, hence 300
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
    # obtain index with the maximum probability, representing the predicted category 
    _, categoryOutput = torch.max(categoryOutput.data, 1)
    
    # sigmoid output is rounded to 0 or 1, representing a bad or good rating, respectively
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
        
        # Bi-directional LSTM 
        self.lstm = tnn.LSTM(input_size=word_len,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.drop = tnn.Dropout(p=0.5)
        
        """
        Bi-LSTM + Attention is attempted but did not yield significant 
        improvements in accuracy, and it took a very long time. 
        Hence, a Bi-LSTM model is used instead
        """
        # self.attention = Attention(lstm_hidden_size*2, batch_first=True)

        self.rating_fc = tnn.Linear(2*lstm_hidden_size, 1)
        self.category_fc = tnn.Linear(2*lstm_hidden_size, 5)

    def forward(self, input, length):
        
        """
        Packing the inputs enables the LSTM model to ignore the padded elements,
        therefore not calculating gradients for the padded values 
        during backpropagation
        """
        
        x = pack_padded_sequence(input, length, batch_first=True) 
        
        x, (hidden, cn) = self.lstm(x)

        """
        The forward network of the last LSTM layer (hidden[-1, :, :]) 
        contains information about previous inputs, 
        whereas the backward network (hidden[-2, :, :]) 
        contains information about following inputs.
        We take the last hidden state of the forward output and 
        the last hidden state of the backward output and merge them together.
        """
        
        x_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        x = self.drop(x_cat) # To prevent overfitting
        
        """
        The same input (x) is shared for both ratings and category 
        """
        
        # Rating FC Layer
        rating_out = self.rating_fc(x)
        rating_out = torch.squeeze(rating_out) # remove dim with 1  
        rating_out = torch.sigmoid(rating_out) 
        
        # Category FC Layer
        cat_out = self.category_fc(x)        
        cat_out = F.relu(cat_out) 
        
        return rating_out, cat_out
        
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        
        """
        BCELoss is good for binary classification --- ratings 
        Cross Entropy Loss is good for multi-class classification --- category
        Both losses are added to get the total loss in the model
        """
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


     
