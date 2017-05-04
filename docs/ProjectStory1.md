---
title: System Architecture
layout: default
---
[back](./)

In this project, we are working on the robot that identifies unique speaker locations and voices. We hope to achieve the speaker recognition goal by training Hidden Markov Models for different individuals and identifying people by running the audio through these models. Our robot will have two main components in its functioning: audio localization and speaker recognition. Audio localization involves hearing audio through two microphones and locating the person by using geometry of the mics and differences in the sound received by each mic. Speaker recognition makes use of Hidden Markov Models to extract and analyze features of each person.

## System Architecture

![System Architecture]()

Our speaker recognition system is organized into three ROS nodes.  The first node is responsible for both publishing the audio that is received via two microphones on a Raspberry Pi and for interacting with the user, who can choose to either train a new speaker or to start a conference call and attempt to recognize voices.  The second node is for audio localization, which subscribes to the “audio” ROS topic, calculates the position of the current speaker in three-dimension space, and controls the robot so that it faces the speaker.  The last node is for training and recognizing speakers, which it does using Mel-frequency cepstral coefficients (MFCC) and hidden Markov models (HMM).

We are using two microphones connected to a Raspberry Pi to record and input audio for audio localization. By using two microphones, we will calculate the Interaural Time Difference (ITD)  of a person’s speech. This time difference between a sound being picked up by each microphone can be used to find the direction of the sound in space. By filtering the incoming audio prior this step, it can distinguish speakers from the rest of the background noise. We did find an existing library that does audio localization from two microphones mounted on a roomba that we can connect back with theory in implementation.

![Training Workflow]()

MFCCs are used widely in speaker recognition problems, and their purpose in this context is to extract features from audio that are well-suited for uniquely identifying a human voice.  We will use the [python_speech_features](https://github.com/jameslyons/python_speech_features) library to extract voice features from both our training dataset and our live conference calls.  This involves splitting the audio in frames of 20 ms, calculating the power spectrum of each frame, and applying a mel filterbank to the power spectra to calculate the energy in each filter.  The result is a set of MFCC coefficients for each frame.  We then want to compress this data so that we are left with only the key information needed for speaker identification.  To do so, we use vector quantization to cluster these frames around a smaller number of centroid points, using with the K-means algorithm.  These centroid points are given labels and compiled into a codebook that represents the speaker in question.  These centroid points, or voice features, represent an observation sequence that is used to calculate the parameters for a hidden Markov model for the speaker.

![Testing Workflow]()

Given this set of features extracted from particular speaker’s voice, we train a hidden Markov model (HMM) to represent the speaker so that he or she can be identified during a conference call.  An HMM is composed for a set of states Q, a transition probability matrix A, a sequence of observations O, and a sequence of emission probabilities B.  Training a model involves calculating the A and B probability matrices for a particular speaker.  Recognizing a speaker then involves determining which existing speaker model yields the highest probability of corresponding to the live speaker audio.
An HMM computation involves three algorithms, the forward algorithm, the Viterbi algorithm, and the forward-backward algorithm, which are explained in more detail below.  The sequence of observations is the set of MFCC features discussed above, which will be input to the forward algorithm.

## Hidden Markov Models

HMMs can be used to emulate a system that have finite hidden states that generate a set of events (observations), where the current state is dependent only on the previous state.

A very popular (and trivial) example that could use HMM is weather prediction using ice-cream consumption. The problem is that you have access to a 10-year old Brook’s diary where she records the number of ice-creams she eats everyday. The assumption here is that there is some correlation between the weather that day (Hot or Cold) and the ice-cream consumption (one, two, etc.). We also assume that the weather on a particular day is dependent only on the weather of the day before.

![Ice-cream Example]()

There are three probabilities that are needed to define an HMM (along with the states) to keep in mind:

  1. Start probability: Chances of entering the Hot or Cold states.
  2. Emission probability: Chances of an observation occurring in a state. How likely will Brook eat 3 ice-creams on a Hot day?
  3. Transmission probability: Chances of going from one state to another. Chances of going from a Hot day to a Cold day.
  
These three are usually symbolised with λ. In order to learn the λ, there are three steps and corresponding algorithms for the steps.

### Forward

Given an HMM with λ and a set of observations (for ex. 3, 2, 2), determine the likelihood of seeing an observation. What is the probability that Brook will consume 3, 2, 2 ice-creams on 3 consecutive days? We will consider all the different combinations of weather (Hot Hot Cold, Cold Hot Cold, etc.) and calculate the probability of 3, 2, 2 and add up all the calculated probabilities.

## Viterbi

Given an HMM with λ and a set of observations (for ex. 3, 2, 2), calculate the best hidden state sequence. We want to know what sequence (something like Hold Cold Hot) is the most likely for the ice-creams eaten by Brook (3, 2, 2 ice-creams). 

## Forward-Backward

GIven a set of observations and a set of states in HMM, learn the best value for the probabilities that make up λ. Here, we have lots of data on Brook’s ice-cream consumption and set of weather pattern, and we are trying to predict the start, transition and emission probabilities of the HMM.

## User Interface

The user-facing component of this project is a command-line interface. Once initialized, it walks the user through the process. It asks for the first person’s name and requests them to speak. Once, there is significant data, the program asks if there are more people present. If yes, it repeats the process until all people’s voices have been entered. It can now enter the recognizing phase. Whenever a voice speaks, it computes probabilities and prints out the most-likely person’s name.

## Current Status of the Project

So far, we have spent a lot of time reading papers to understand the math behind HMM, MFCC, and audio localization. We have written out the Forward and Viterbi algorithms in python to start implementing HMM computationally. We have also begun writing code to implement MFCC, including the doing VQ with k-means. We have had issues collecting audio up until now because of a faulty RasPi. We now know how to successfully set up the RasPi but are still waiting to reflash it so we can actually collect audio data.

[back](./)
