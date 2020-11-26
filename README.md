# Business Reviews - the Good, the Bad and the Many 

## Aim 
1. To determine if a business review is positive or negative 
2. To categorise the review into one of the 5 classes: 0=Restaurants, 1=Shopping, 2=Home Services, 3=Health & Medical, 4=Automotive

## Input
The dataset consists of 50,000 reviews from different categories and sentiment.
Each review comes in the following JSON format:

`{"businessCategory": 0, "rating": 1, "reviewText": "By far the cleanest restaurant I've been to (including kitchen). The owner is a fantastic guy and always very attentive to the needs of the customers. Food is quite good and consistent."}`

## LSTM Model 
Before developing the model, the review text is preprocessed by removing punctuations and replacing specific symbols. Stopwords are also added by referring to the NLTK corpus. The review text is tokenized and embedded using GloVe with a vector dimension of 300. For this model, a Bidirectional LSTM is chosen because they are capable of learning long-term dependencies, hence suitable for analysing a long review text. The vector embeddings are passed through the Bi-LSTM layer. Then, the last hidden state of the Bi-LSTM layer is passed into two different fully connected (linear) layers, followed by their respective activation functions, to generate two output tensors - rating and category. To convert the network’s output to the predicted labels, rating output is rounded to 0 or 1, whereas the index of the highest value in the category tensor is selected.  For the loss function, binary cross entropy is assigned for rating whereas cross entropy is chosen for category. Both loss functions are ideal candidates for classification tasks. 

Initially, a LSTM model is developed as a base model. Then, several design improvements are incorporated into the base model and have proven to work, including:
-   Using a bidirectional LSTM instead of uni-directional to obtain better context of the review text
-   Replacing SGD with Adam Optimiser, with a learning rate of 0.001
-   Increasing word vector size from 50 to 300 to represent the semantics better
-   Adding Dropout layer to prevent overfitting 
-   Preprocessing Data

However, some methods were unable to boost the accuracy, such as:
-   Having more than 1 Bi-LSTM layer
-   Adding more Linear layers
-   Word Lemmatization 

## LSTM + Attention
This model is similar to the LSTM model aforementioned, but with an additional attention layer. Incorporating an Attention Layer increased the score by 1% but consumed a lot of computational time and resources. Hence, LSTM model is chosen to train the data based on Ockham Razor's theorem,.

## Results
Overall, the LSTM model successfully achieved the aim because it consistently obtained a weighted score of 85% within 5 epochs.

**Code written by: UNSW, Ming Xuan CHUA and Qie Shang PUA**
