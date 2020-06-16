Separable Bandit Classification
===============================

This repository contains source codes of the experiments conducted in the paper titled _Bandit Multiclass
Linear Classification: Efficient Algorithms for the Separable Case_ authored by
[Alina Beygelzimer](http://hunch.net/~beygel/),
[Dávid Pál](http://david.palenica.com/),
[Balázs Szörényi](http://www.inf.u-szeged.hu/~szorenyi/),
[Devanathan Thiruvenkatachari](https://cims.nyu.edu/~deva/),
[Chen-Yu Wei](https://sites.google.com/site/bobcywei/)
and [Chicheng Zhang](https://zcc1307.github.io/).
The paper was accepted at [ICML 2019](https://icml.cc/Conferences/2019).

Bandit multiclass classification is a problem where in each round an algorithm
receives a feature vector and it needs to predict one of <b>K</b> classes, after
which it receives binary feedback whether or not the class was correct. We
design efficient algorithms under the assumption that data are linearly
separable.
