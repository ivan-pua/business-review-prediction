#!/usr/bin/env python3
"""
Main file to train and test model

UNSW COMP9444 Neural Networks and Deep Learning

"""

import torch
from torchtext import data

from config import device
import lstm_attention_model as lstm_model

def main():
    print("Using device: {}"
          "\n".format(str(device)))

    # Load the training dataset, and create a dataloader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           tokenize=lstm_model.tokenise,
                           preprocessing=lstm_model.preprocessing,
                           postprocessing=lstm_model.postprocessing,
                           stop_words=lstm_model.stopWords)
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

    dataset = data.TabularDataset('train.json', 'json',
                                 {'reviewText': ('reviewText', textField),
                                  'rating': ('rating', labelField),
                                  'businessCategory': ('businessCategory', labelField)})

    textField.build_vocab(dataset, vectors=lstm_model.wordVectors)

    # Allow training on the entire dataset, or split it for training and validation.
    if lstm_model.trainValSplit == 1:
        trainLoader = data.BucketIterator(dataset, shuffle=True,
                                          batch_size=lstm_model.batchSize,
                                          sort_key=lambda x: len(x.reviewText),
                                          sort_within_batch=True)
    else:
        train, validate = dataset.split(split_ratio=lstm_model.trainValSplit)

        trainLoader, valLoader = data.BucketIterator.splits((train, validate),
                                                            shuffle=True,
                                                            batch_size=lstm_model.batchSize,
                                                            sort_key=lambda x: len(x.reviewText),
                                                            sort_within_batch=True)

    # Get model and optimiser from lstm_model.py
    net = lstm_model.net.to(device)
    lossFunc = lstm_model.lossFunc
    optimiser = lstm_model.optimiser

    # Train.
    for epoch in range(lstm_model.epochs):
        runningLoss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
            length = batch.reviewText[1].to(device)
            rating = batch.rating.to(device)
            businessCategory = batch.businessCategory.to(device)

            # PyTorch calculates gradients by accumulating contributions to them
            # (useful for RNNs).  Hence we must manually set them to zero before
            # calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            ratingOutput, categoryOutput = net(inputs, length)
            loss = lossFunc(ratingOutput, categoryOutput, rating, businessCategory)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            runningLoss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                      % (epoch + 1, i + 1, runningLoss / 32))
                runningLoss = 0

    # Save model.
    torch.save(net.state_dict(), 'savedModel.pth')
    print("\n"
          "Model saved to savedModel.pth")

    # Test on validation data if it exists.
    if lstm_model.trainValSplit != 1:
        net.eval()

        correctRatingOnlySum = 0
        correctCategoryOnlySum = 0
        bothCorrectSum = 0
        with torch.no_grad():
            for batch in valLoader:
                # Get a batch and potentially send it to GPU memory.
                inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                length = batch.reviewText[1].to(device)
                rating = batch.rating.to(device)
                businessCategory = batch.businessCategory.to(device)

                # Convert network output to integer values.
                ratingOutputs, categoryOutputs = lstm_model.convertNetOutput(*net(inputs, length))

                # Calculate performance
                correctRating = rating == ratingOutputs.flatten()
                correctCategory = businessCategory == categoryOutputs.flatten()

                correctRatingOnlySum += torch.sum(correctRating & ~correctCategory).item()
                correctCategoryOnlySum += torch.sum(correctCategory & ~correctRating).item()
                bothCorrectSum += torch.sum(correctRating & correctCategory).item()

        correctRatingOnlyPercent = correctRatingOnlySum / len(validate)
        correctCategoryOnlyPercent = correctCategoryOnlySum / len(validate)
        bothCorrectPercent = bothCorrectSum / len(validate)
        neitherCorrectPer = 1 - correctRatingOnlyPercent \
                              - correctCategoryOnlyPercent \
                              - bothCorrectPercent

        score = 100 * (bothCorrectPercent
                       + 0.5 * correctCategoryOnlyPercent
                       + 0.1 * correctRatingOnlyPercent)

        print("\n"
              "Rating incorrect, business category incorrect: {:.2%}\n"
              "Rating correct, business category incorrect: {:.2%}\n"
              "Rating incorrect, business category correct: {:.2%}\n"
              "Rating correct, business category correct: {:.2%}\n"
              "\n"
              "Weighted score: {:.2f}".format(neitherCorrectPer,
                                              correctRatingOnlyPercent,
                                              correctCategoryOnlyPercent,
                                              bothCorrectPercent, score))
        
        # Settings
        print(f"\n----Settings----\nEpochs:{lstm_model.epochs}\nBatch-size:{lstm_model.batchSize}\nTraining Ratio={lstm_model.trainValSplit}\nOptimiser={lstm_model.optimiser}\nWord Vector Dim={lstm_model.word_len}\nBi-LSTM\nNum layers = {lstm_model.lstm_layers}\nHidden Size={lstm_model.lstm_hidden_size}\n")
        
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"The total params are {total_params} and the trainable one are {trainable_params}")

if __name__ == '__main__':
    main()
