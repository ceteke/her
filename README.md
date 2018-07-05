# DQN + HER

This repository contains the implementation of DQN + HER. The implementation is tested on the toy
problem presented in the [paper](https://arxiv.org/pdf/1707.01495.pdf). 
Here is a [blog post](https://becominghuman.ai/learning-from-mistakes-with-hindsight-experience-replay-547fce2b3305) about HER.  

The hyperparameters used in this repo are the same as the paper..  
* $\alpha$: 0.001
* $\gamma$: 0.98
* Q-Network is an MLP with 256 hidden units
* Buffer holds up to $10^6$ transitions

### How to train?
```python train.py --help```  
```
usage: train.py [-h] [-v] [-s S] [-i I] [-e E] [-c C] [-o O]

HER Bit Flipping

optional arguments:
  -h, --help  show this help message and exit
  -v          Verbose flag
  -s S        Size of bit string
  -i I        Num epochs
  -e E        Num episodes
  -c C        Num cycles
  -o O        Optimization steps

```

### Inference
TODO

### Results
TODO