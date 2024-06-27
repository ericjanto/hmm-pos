## PoS Tagging with HMMs
This project implemented and trained a [Hidden-Markov-Model (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model) for tagging the [part-of-speech (PoS)](https://en.wikipedia.org/wiki/Part_of_speech) in English sentences. This was achieved by estimating the emission model which emits the PoS tags, and further estimating the transition model for probabilities from one state to another (where states are defined as (word,tag) pairs).

The Viterbi algorithm, a dynamic programming algorithm, was implemented to train the HMM.

## Setup
You can create a virtual env using conda. For this you need
anaconda installed in your system. A recommended configuration
is to set the automatic activation of the `base` env to false,
running:

```
conda config --set auto_activate_base False
```

Using the `enviroment.yml` you can create the new env.

1. Create the environment using:
   ```
   conda env create -f environment.yml
   ```
2. Activate the environment
   ```
   conda activate hmm
   ```


Update / add packages:

1. conda activate fnlp
2. conda install --name hmm <package> -y
3. conda env export > environment.yml
