```bibtex
@misc{moe-infinity,
  author       = {Leyang Xue and
                  Yao Fu and
                  Zhan Lu and
                  Chuanhao Sun and
                  Luo Mai and
                  Mahesh Marina},
  title        = {MoE-Infinity: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache},
  archivePrefix= {arXiv},
  eprint       = {2401.14361},
  year         = {2024}
}
```

The optional CPU expert kernels (`extensions/kernel/cpu/sglang/`, used by
`moe_infinity/kernel/cpu/`) are SGLang's x86 CPU MoE/GEMM kernels
(https://github.com/sgl-project/sglang, Apache-2.0, SGLang Team); see the
NOTICE file in that directory for the pinned commit.

The optional BatchGen expert kernels (`moe_infinity/kernel/batchgen/`) come
from BatchGen; please also cite it when using them:

```bibtex
@inproceedings{batchgen-osdi26,
  author    = {Tairan Xu and Leyang Xue and Zhan Lu and Jinfu Deng and
               Hongyang Xiao and Yinsicheng Jiang and Congjie He and
               Matej Sandor and Le Xu and Luo Mai},
  title     = {BatchGen: An Architecture for Scalable and Efficient Batch Inference},
  booktitle = {20th USENIX Symposium on Operating Systems Design and
               Implementation (OSDI 26)},
  pages     = {1125--1141},
  year      = {2026},
  url       = {https://www.usenix.org/conference/osdi26/presentation/xu-tairan}
}
```
