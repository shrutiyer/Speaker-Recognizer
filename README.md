To validate our HMM functions, we tested our system with what is known as the ice cream problem.  One of the advantages of the ice cream problem is that both its observations and its hidden states are clearly defined.  

The concept is that you would like to predict weather patterns (sequence of hidden states) using your ice-cream consumption (observations).  You keep a diary of how many ice creams you eat everyday, and you make the assumption that there is some correlation between the weather that day (hot or cold) and your ice cream consumption.  

You also make the assumption that the weather on a particular day is dependent only on the weather of the day before.  The transition probabilities represent the chances of going from, for example, a hot day to a cold day.  The emission probabilities represent the chance of, for example, you eating 3 ice creams on a hot day.  

We referenced [this implementation](http://cs.jhu.edu/~jason/papers/eisner.hmm.xls) of a hidden Markov Model using the ice cream problem, discussed in [Jason Eisner’s](https://cs.jhu.edu/~jason/papers/eisner.tnlp02.pdf) paper, to verify that our hidden Markov model produced the same results.

To run the code, simply:

`python recognizer.py`

