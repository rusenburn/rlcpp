# Reinforcement Learning C++ ( RLCPP )

## Introduction
* RLCPP project uses a model-based training that is similar to alpha-zero to teach an agent to play perfect information 2 players zero-sum environments, supported environments are :

    * Tic Tac Toe
    * Othello
    * Damma
    * English Draughts
    * Walls ( custom environment did not know the original name )

## Requirements
* MSVC Compiler ( on windows )
* cmake
* CUDA 11.8 ( If you have cuda device)
* cuDNN tools (If you have cuda device) (v8.5.0.96 used)
* Libtorch 2.2.1