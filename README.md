# Reproduction notes (Tadhg Roche, August 2025)

This repository is a fork of the official OPQN code by Ming Zhang, Xuefei Zhe and Hong Yan
(*Orthonormal Product Quantization Network for Scalable Face Image Retrieval*, Pattern Recognition, 2023).
Everything below this section is their README, kept as published. My aim was to get the method running
end to end on VGGFace2 and check that it reaches the kind of retrieval performance the paper reports.

## What I changed

- **Fixed a crash on VGGFace2:** `data_loader.py` referenced `datasets` instead of `dataset`, so the VGGFace2
  branch never ran. Also renamed the cross-dataset flag (`args.c` → `args.cross_dataset`) and added a
  `--data-dir` argument so the dataset can live outside the repo.
- **Rewrote the per-sample transforms** in the data loader so that every image is resized/cropped to a fixed size
  before batching (the original pipeline produced mismatched tensor sizes on this data).
- **Built a leak-free split:** a per-identity train/test split of a 480-identity VGGFace2 subset
  (141,308 training and 35,090 test images), with every identity present in both splits and no image in both.

## What I got

- 64-bit model (8 codebooks × 256 words), batch size 256, lr 0.1, 160 epochs on a single cloud GPU
  (about 72 s per epoch): **mAP 97.27%, top-k precision 96.92%** on the held-out split.
- A small retrieval demo (`notebooks/`) that loads the checkpoint and returns the nearest faces for a query image.
- An attempt at the full official split (709,662 training images, 2,787 identities) at 36 and 64 bits ran at
  25–30 minutes per epoch and did not converge in the budget I had (`log/`); the first author kindly shared a
  trained VGGFace2 checkpoint and the supplementary material by email.

## Running the reproduction

The Colab notebook in `notebooks/OPQN_vggface2_reproduction.ipynb` contains the full sequence: clone, patch,
download the subset from Kaggle, split, train, evaluate, and demo. The trained checkpoint (243 MB) is not
committed; ask me for it.

---

# Orthonormal Product Quantization Network for Scalable Face Image Retrieval
Pytorch implementation of "[Orthonormal Product Quantization Network for Scalable Face Image Retrieval](https://arxiv.org/abs/2107.00327) (OPQN)".

[Official version](https://doi.org/10.1016/j.patcog.2023.109671) is published in Pattern Recognition. Supplementary material can be found [there](https://drive.google.com/file/d/1XsmCeykToR8FFlSKi3D3vOK_BCa4Ekkd/view?usp=sharing). 

# Introduction
OPQN is a novel deep quantization method that produces compact binary codes for large-scale face image retrieval. The method employs **predefined orthonormal codewords** to increase quantization informativeness while reducing codeword redundancy. To maximize discriminability among identities in each quantization subspace for both the quantized and original features, a tailored loss function is utilized. Extensive experiments conducted on commonly-used face image datasets demonstrate that OPQN achieves state-of-the-art performance. The proposed orthonormal codewords consistently enhance both the models' standard retrieval performance and generalization ability. Furthermore, the method can also be applied to general image retrieval tasks, showcasing the broad superiority of the proposed codewords scheme and the applied learning metric. Overall, the proposed method provides a general framework for deep quantization-based image retrieval.

# Overall training procedure of OPQN
<img src="/figures/overview.png" alt="drawing" width="85%"/>
<p></p>

# Results
<img src="/figures/pr_three_all2.png" alt="drawing" width="100%"/>
<p></p>
<center> Performance measured by precision-recall curves: (a) 48-bit on FaceScrub, (b) 48-bit on CFW-60K, (c) 64-bit on VGGFace2.</center>

# Usage
## Training
Suppose that the dataset is under the base directory `./data` with folderpath `./data/dataset_name`. The below command will train a 36-bit deep product quantizaion model on FaceSrub dataset using 6 codebooks with 64 codewords per codebook, and default hyperparameters.
 ```
python opqn_main.py --dataset facescrub --save your_model_name.tar --len 36 --num 6 --words 64
 ```
 
## Testing
To evaluate a series of models on a dataset with different code lengths, e.g. 16-bit, 24-bit and 36-bit codes, with 2, 4, 6 codebooks and 256, 64, 64 codewords per codebook, respectively: 
```
python opqn_main.py -e --dataset facescrub --load model1.tar model2.tar model3.tar --len 16 24 36 --num 2 4 6 --words 256 64 64
 ```
 Add `-c` for cross-dataset evaluation.
 
 Run ```python opqn_main.py --help``` to see the detailed explanation of each argument.

# Citation
If you find the codes are useful to your research, please consider citing our PR paper:
```
@article{zhang2023orthonormal,
  title={Orthonormal Product Quantization Network for Scalable Face Image Retrieval},
  author={Zhang, Ming and Zhe, Xuefei and Yan, Hong},
  journal={Pattern Recognition},
  pages={109671},
  year={2023},
  publisher={Elsevier}
}
```
# Related Projects
 \[1\] Deep Center-Based Dual-Constrained Hashing for Discriminative Face Image Retrieval [(DCDH)](https://github.com/mzhang367/DCDH-PyTorch)
 
 \[2\] Generalized Product Quantization Network for Semi-supervised Image Retrieval [(GPQ)](https://github.com/youngkyunJang/GPQ)
