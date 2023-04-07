# PAIR-Diffusion
This is the official repo for the work:

**PAIR-Diffusion: Object-Level Image Editing with
Structure-and-Appearance Paired Diffusion Models**<br>
Vidit Goel*<sup>1</sup>, 
Elia Peruzzo*<sup>1,2</sup>, 
Yifan Jiang<sup>3</sup>, 
Dejia Xu<sup>3</sup>, 
Nicu Sebe<sup>2</sup>,
Trevor Darrell<sup>4</sup>,
Zhangyang Wang<sup>1,3</sup>, 
[Humphrey Shi<sup>1,5,6</sup>](https://www.humphreyshi.com/)<br>

In association with Picsart AI Research (PAIR)<sup>1</sup>, UTrento<sup>2</sup>, UT Austin<sup>3</sup>, UC Berkeley<sup>4</sup>, UOregon<sup>5</sup>, UIUC<sup>6</sup>

*equal contribution

We built Structure and Appearance Paired (PAIR) Diffusion that allows reference image-guided appearance manipulation and structure editing of an image at an object level. Describing object appearances using text can be challenging and ambiguous, PAIR Diffusion enables a user to control the appearance of an object using images.

Try our demo at [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/PAIR/PAIR-Diffusion)


[[arXiv](https://arxiv.org/abs/2303.17546)][[Video](https://drive.google.com/file/d/1s96TNWgAwBjRDTyFTWakZosO3Ot_Fny6/view?usp=share_link)]

## News
 * [04/07/2023] Demo relased on [🤗Huggingface space](https://huggingface.co/spaces/PAIR/PAIR-Diffusion)!
 * [03/30/2023] Paper released on [arXiv](https://arxiv.org/abs/2303.17546)


## Results
Given below are results for appearace editing using our method on SDv1.5
<img src="assets/ir.gif" width="99%" raw=true>
<img src="assets/ski.gif" width="99%" raw=true>
<img src="assets/door.gif" width="99%" raw=true>

## Object Level Image Editing
### Stable Diffusion Results
* Applying our method to Stable Diffusion allows reference image based editing in the wild

<p align="center">
  <img src="assets/SD_app.png" width="99%">
</p>

* We can edit both structure and appearance of the objects independently.

<p align="center">
  <img src="assets/SD_Teaserv1.png" width="99%">
</p>


### Unconditional Diffusion Models
* PAIR diffusion also works with unconditional diffusion models, we show results on LSUN churches, bedrooms and CelebA-HQ. We can perform edits using both in-domain and out-of-domain reference images.
<p align="center">
  <img src="assets/Teaser.png" width="99%">
</p>

## BibTeX
If you use our work in your research, please cite our publication:
```
@article{goel2023pair,
      title={PAIR-Diffusion: Object-Level Image Editing with Structure-and-Appearance Paired Diffusion Models},
      author={Goel, Vidit and Peruzzo, Elia and Jiang, Yifan and Xu, Dejia and Sebe, Nicu and Darrell, Trevor and 
      Wang, Zhangyang and Shi, Humphrey},
      journal={arXiv preprint arXiv:2303.17546},
      year={2023} 
      }
```