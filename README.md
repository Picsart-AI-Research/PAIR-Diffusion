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

[[arXiv]()][[Video](https://drive.google.com/file/d/1s96TNWgAwBjRDTyFTWakZosO3Ot_Fny6/view?usp=share_link)]

## Object Level Editing Results
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

## More Results
We show more results for appearace editing using our method on SDv1.5
<img src="assets/ir.gif" width="99%" raw=true>
<img src="assets/ski.gif" width="99%" raw=true>
<img src="assets/door.gif" width="99%" raw=true>
<!-- <center>
<table class="center">
<tr align="center" >
  <td style="font-size: 20px;">Input Image</td>
  <td style="font-size: 20px;">Reference Image</td>
  <td style="font-size: 20px;">Edited Image</td>              
</tr>
<tr align="center">
  <td><img src="assets/individual_images/ironmanv2.jpg" height="220"></td>
  <td><img src="assets/individual_images/hulk.jpeg" height="220"></td>
  <td><img src="assets/individual_images/ironandhuktogether.png" height="220"></td>              
</tr>
<tr align="center">
  <td></td>
  <td></td>
  <td><i>A picture of hulk and ironman </i></td>              
</tr>  
<tr align="center">
  <td><img src="assets/individual_images/ski.jpeg" height="250"></td>
  <td><img src="assets/individual_images/lava.jpeg" height="250"></td>
  <td><img src="assets/individual_images/skiedit.png" height="250" style="padding-top: 36px;"></td>              
</tr>  
  <tr align="center">
  <td></td>
  <td></td>
  <td><i>A person skiing</i></td>              
</tr>  
<tr align="center">
  <td><img src="assets/individual_images/metaldoor.jpeg" height="250"></td>
  <td><img src="assets/individual_images/door.jpeg" height="250"></td>
  <td><img src="assets/individual_images/editdoor1.png" height="250" style="padding-top: 36px;"></td>              
</tr>  
<tr align="center">
  <td></td>
  <td></td>
  <td><i>Picture of a door</i></td>              
</tr>  
</table> -->
