# RNAM: Recurrent Neural Additive Model

A batched implementation of an affine/linear transformation and the minGRU as proposed in ["Were RNNs All We Needed?"](https://arxiv.org/pdf/2410.01201v3).
Useful for neural additive/ensemble modeling of tabular and longitudinal data.

## Attribution

The initialization used in [linear.py](rnam/linear.py) is adopted from the [PyTorch nn.Linear initialization](https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/linear.py#L117) ([BSD-3-Clause license](licenses/pytorch.BSD3)).
The code in [gru.py](rnam/gru.py) is adapted from
[Phil Wang's implementation of the minGRU](https://github.com/lucidrains/minGRU-pytorch/blob/9fe95d623b2a30f5cbc689e4640dc62403da0df5/minGRU_pytorch/minGRU.py) ([MIT License](licenses/lucidrains.MIT)), which itself incorporates [Franz Heinsen's log-space implementation of the parallel scan algorithm](https://github.com/glassroom/heinsen_sequence/blob/b747964a6fda7048558791e0d29060fe69035507/README.md) ([MIT License](licenses/glassroom.MIT)).
The minGRU was proposed in ["Were RNNs All We Needed?"](https://arxiv.org/pdf/2410.01201v3), which simplifies the GRU to a minimal variant that can be trained efficiently in parallel.
The implemented cell uses the parallel scan proposed in ["Efficient Parallelization of a Ubiquitous Sequential Computation"](https://arxiv.org/pdf/2311.06281v4).

License texts are available [here](licenses/).<br>
Last accessed: 2025-12-13

## Install

```bash
$ pip install git+https://github.com/kachelriess/rnam
```

## References

```bibtex
@inproceedings{Feng2024WereRA,
    title   = {Were RNNs All We Needed?},
    author  = {Leo Feng and Frederick Tung and Mohamed Osama Ahmed and Yoshua Bengio and Hossein Hajimirsadegh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273025630}
}
```

```bibtex
@misc{heinsen2023efficientparallelizationubiquitoussequential,
    title   = {Efficient Parallelization of a Ubiquitous Sequential Computation},
    author  = {Franz A. Heinsen},
    year    = {2023},
    url     = {https://arxiv.org/abs/2311.06281v4},
}
```

## Citation

```bibtex
@misc{rnam2025,
    title   = {RNAM: Recurrent Neural Additive Model},
    author  = {Kachelrieß, Lucas},
    year    = {2025},
    note    = {Software available at \url{https://github.com/kachelriess/rnam}}
}
```
