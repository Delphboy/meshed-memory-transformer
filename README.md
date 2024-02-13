# M²: Meshed-Memory Transformer Fork
- Fixed the `/` vs `//` error in the original code
- Improved speed of the `Dataset` class
- **Original Code available:** [https://github.com/aimagelab/meshed-memory-transformer](https://github.com/aimagelab/meshed-memory-transformer)
- Added `hpc` scripts and `setup.qsub`

## M²: Meshed-Memory Transformer
This repository contains the reference code for the paper _[Meshed-Memory Transformer for Image Captioning](https://arxiv.org/abs/1912.08226)_ (CVPR 2020).

Please cite with the following BibTeX:

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

## Environment setup
See `setup.qsub`. On QMUL's Apocrita/Andrena hpc system, this job can be automated with the following steps:

1. Change `$USER` to your username
2. Check the directories are as expected
3. Run `qsub setup.qsub`


## Training procedure
See `train.py` for the complete list of arguments. An hpc system script has been provided in `hpc/train.qsub`. **Ensure the script is ammeded to account for your username and directory structure.** Submit the job with `qsub train.qsub` from within the `hpc` directory.

## Results
<p align="center">
  <img src="images/results.png" alt="Sample Results" width="850"/>
</p>

#### References
[1] P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, 2018.
