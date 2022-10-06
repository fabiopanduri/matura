# Matura: Source Code

This is the source code to the matura paper
"Two ex-nihilo Implementations of Reinforcement Learning Algorithms"
by Luis Hartmann and Fabio Panduri.

It implements the algorithms Deep Q-Learning and NeuroEvolution
of Augmenting Topologies as published by Mnih et al. (2015)
and Stanley (2002).
This also includes an ex-nihilo SGD implementation.
It features functions to evaluate the algorithm's performances
by having them play various games, as well as to produce
statistical data and plots from the data.

Currently implemented game environments:
- cartpole
- pong

## Requirements
Install required modules with ``pip install -r requirements.txt``.
Other versions of modules may not work.
Python version known to work is 3.9.2. Other versions should
not cause many issues.

## Usage
To run the algorithms the ``main.py`` file must be run with python3.
The first argument must specify the algorithm (dql, neat, sgd).
After having specified the algorithm optional arguments may be added.
The optional arguments are most likely different depending on the algorithm
specified.
The command will then look something like
``python3 main.py algorithm --optional-parameters``
To receive a list of all available optional parameters concerning a specific algorithm type
``python3 main.py the-algorithm -h``
So for example
``python3 main.py neat -h``
will print a list of all available neat flags.

## Examples
``python3 main.py dql -g pong -e 300 --reward-system v3``
runs 300 iterations of DQL on pong using reward system v3.
``python3 main.py neat -g cartpole -i 200``
runs 200 episodes of NEAT on cartpole.
``python3 main.py sgd -e 1000 -b 1000``
runs 1000 epochs of sgd with a batch size of 1000.


## Bibliography (BibTeX)
```
@Article{mnih_nature15b,
	author = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and Petersen, Stig and Beattie, Charles and Sadik, Amir and Antonoglou, Ioannis and King, Helen and Kumaran, Dharshan and Wierstra, Daan and Legg, Shane and Hassabis, Demis}, 
	title = {Human-level control through deep reinforcement learning},
	journal = {Nature},
	volume = {518},
	pages = {529--533},
	year = {2015},
	publisher = {Nature Publishing Group, a division of Macmillan Publishers Limited. All Rights Reserved.},
	DOI = {doi:10.1038/nature14236},
}

@Article{stanley_neat-ec02,
	author = {Stanley, Kenneth O. and Miikkulainen, Risto},
	title = {Evolving Neural Networks through Augmenting Topologies},
	journal = {Evolutionary Computation},
	volume = {10},
	number = {2},
	pages = {99--127},
	year = {2002},
	publisher = {The MIT Press Journals},
}
```
