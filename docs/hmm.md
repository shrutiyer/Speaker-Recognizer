## Hidden Markov Models

HMMs can be used to emulate a system that have finite hidden states that generate a set of events (observations), where the current state is dependent only on the previous state.

A very popular (and trivial) example that could use HMM is weather prediction using ice-cream consumption. The problem is that you have access to a 10-year old Brook’s diary where she records the number of ice-creams she eats everyday. The assumption here is that there is some correlation between the weather that day (Hot or Cold) and the ice-cream consumption (one, two, etc.). We also assume that the weather on a particular day is dependent only on the weather of the day before.

![Ice-cream Example](/images/HMM Hot Cold.jpg)

There are three probabilities that are needed to define an HMM (along with the states) to keep in mind:

  1. Start probability: Chances of entering the Hot or Cold states.
  2. Emission probability: Chances of an observation occurring in a state. How likely will Brook eat 3 ice-creams on a Hot day?
  3. Transmission probability: Chances of going from one state to another. Chances of going from a Hot day to a Cold day.
  
These three are usually symbolised with λ. In order to learn the λ, there are three steps and corresponding algorithms for the steps.

### Forward

Given an HMM with λ and a set of observations (for ex. 3, 2, 2), determine the likelihood of seeing an observation. What is the probability that Brook will consume 3, 2, 2 ice-creams on 3 consecutive days? We will consider all the different combinations of weather (Hot Hot Cold, Cold Hot Cold, etc.) and calculate the probability of 3, 2, 2 and add up all the calculated probabilities.

### Viterbi

Given an HMM with λ and a set of observations (for ex. 3, 2, 2), calculate the best hidden state sequence. We want to know what sequence (something like Hold Cold Hot) is the most likely for the ice-creams eaten by Brook (3, 2, 2 ice-creams). 

### Forward-Backward

Given a set of observations and a set of states in HMM, learn the best value for the probabilities that make up λ. Here, we have lots of data on Brook’s ice-cream consumption and set of weather pattern, and we are trying to predict the start, transition and emission probabilities of the HMM.
