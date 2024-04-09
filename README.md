# M²: Meshed-Memory Transformer Fork
- **Original Code available:** [https://github.com/aimagelab/meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer)

## Fork Contributions
- Fixed the `/` vs `//` error in the original code [following this discussion](https://github.com/aimagelab/meshed-memory-transformer/issues/82)
- Improved speed of the `Dataset` class
- Added `hpc` scripts and `setup.qsub`
- Added loss/eval training and validation plot functionality (runs automatically)
- Added a potential fix to the [`<eos>` bug in SCST](https://github.com/aimagelab/meshed-memory-transformer/issues/46).

## $\mathcal{M}^2$: Meshed-Memory Transformer
This repository contains a fork of the reference code for the paper _[Meshed-Memory Transformer for Image Captioning](https://arxiv.org/abs/1912.08226)_ (CVPR 2020).

Please cite with the original work BibTeX:

```
@inproceedings{cornia2020m2,
  title={{Meshed-Memory Transformer for Image Captioning}},
  author={Cornia, Marcella and Stefanini, Matteo and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
<p align="center">
  <img src="images/m2.png" alt="Meshed-Memory Transformer" width="320"/>
</p>

## Setup

### SPICE Evaluations

Run the following:
```bash
cd evaluations
bash get_stanford_models.sh
```

See [this post](https://henrysenior.com/words/2024-04-03-adding-spice-to-meshed-memory) for more information.

### Environment Setup
See `setup.qsub`. On QMUL's Apocrita/Andrena hpc system, this job can be automated with the following steps:

1. Check the directories are as expected
2. Run `qsub setup.qsub`


## Training procedure
See `train.py` for the complete list of arguments. An hpc system script has been provided in `hpc/train.qsub`. **Ensure the script is ammeded to account for your username and directory structure. i.e. Don't use `$USER$` in the header information.** Submit the job with `qsub train.qsub` from within the `hpc` directory.

## Results
<p align="center">
  <img src="images/results.png" alt="Sample Results" width="850"/>
</p>

#### References
[1] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2018.
