

import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import subprocess
import shlex

from huggingface_hub import hf_hub_url, hf_hub_download
from share import *

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.OneFormer import OneformerSegmenter
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSamplerSpaCFG
from ldm.models.autoencoder import DiagonalGaussianDistribution

urls = {
    'shi-labs/oneformer_coco_swin_large': ['150_16_swin_l_oneformer_coco_100ep.pth'],
    'PAIR/PAIR-diffusion-sdv15-coco-finetune': ['pair_diffusion_epoch62.ckpt']
}

WTS_DICT = {

}

if os.path.exists('checkpoints') == False:
    os.mkdir('checkpoints')
for repo in urls:
    files = urls[repo]
    for file in files:
        url = hf_hub_url(repo, file)
        name_ckp = url.split('/')[-1]

        WTS_DICT[repo] = hf_hub_download(repo_id=repo, filename=file)

print(WTS_DICT)
apply_segmentor = OneformerSegmenter(WTS_DICT['shi-labs/oneformer_coco_swin_large'])

model = create_model('./configs/sap_fixed_hintnet_v15.yaml').cpu()
model.load_state_dict(load_state_dict(WTS_DICT['PAIR/PAIR-diffusion-sdv15-coco-finetune'], location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSamplerSpaCFG(model)
_COLORS = []
save_memory = False

def gen_color():
    color = tuple(np.round(np.random.choice(range(256), size=3), 3))
    if color not in _COLORS and np.mean(color) != 0.0:
        _COLORS.append(color)
    else:
        gen_color()


for _ in range(300):
    gen_color()


class ImageComp:
    def __init__(self, edit_operation):
        self.input_img = None
        self.input_pmask = None
        self.input_segmask = None

        self.ref_img = None
        self.ref_pmask = None
        self.ref_segmask = None

        self.H = None
        self.W = None
        self.baseoutput = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.edit_operation = edit_operation

    def init_input_canvas(self, img):
        img =  HWC3(img)
        img = resize_image(img, 512)
        detected_mask = apply_segmentor(img, 'panoptic')[0]
        detected_seg = apply_segmentor(img, 'semantic')

        self.input_img = img
        self.input_pmask = detected_mask
        self.input_segmask = detected_seg
        self.H = img.shape[0]
        self.W = img.shape[1]
        
        detected_mask = detected_mask.cpu().numpy()

        uni = np.unique(detected_mask)
        color_mask = np.zeros((detected_mask.shape[0], detected_mask.shape[1], 3))
        for i in uni:
            color_mask[detected_mask == i] = _COLORS[i]

        output = color_mask*0.8 + img * 0.2
        self.baseoutput = output.astype(np.uint8)
        return self.baseoutput
    
    def init_ref_canvas(self, img):
        img =  HWC3(img)
        img = resize_image(img, 512)
        detected_mask = apply_segmentor(img, 'panoptic')[0]
        detected_seg = apply_segmentor(img, 'semantic')

        self.ref_img = img
        self.ref_pmask = detected_mask
        self.ref_segmask = detected_seg

        detected_mask = detected_mask.cpu().numpy()

        uni = np.unique(detected_mask)
        color_mask = np.zeros((detected_mask.shape[0], detected_mask.shape[1], 3))
        for i in uni:
            color_mask[detected_mask == i] = _COLORS[i]

        output = color_mask*0.8 + img * 0.2
        self.baseoutput = output.astype(np.uint8)
        return self.baseoutput

    def process_mask(self, mask, panoptic_mask, segmask):
        panoptic_mask_ = panoptic_mask + 1
        mask_ = resize_image(mask['mask'][:, :, 0], min(panoptic_mask.shape))
        mask_ = torch.tensor(mask_)
        maski = torch.zeros_like(mask_).cuda()
        maski[mask_ > 127] = 1
        mask = maski * panoptic_mask_
        unique_ids, counts = torch.unique(mask, return_counts=True)
        mask_id = unique_ids[torch.argmax(counts[1:]) + 1]
        final_mask = torch.zeros(mask.shape).cuda()
        final_mask[panoptic_mask_ == mask_id] = 1

        obj_class = maski * (segmask + 1)
        unique_ids, counts = torch.unique(obj_class, return_counts=True)
        obj_class = unique_ids[torch.argmax(counts[1:]) + 1] - 1
        return final_mask, obj_class
    
    
    def edit_app(self, input_mask, ref_mask, whole_ref):
        input_pmask = self.input_pmask
        input_segmask = self.input_segmask

        if whole_ref:
            reference_mask = torch.ones(self.ref_pmask.shape).cuda()
        else:
            reference_mask, _ = self.process_mask(ref_mask, self.ref_pmask, self.ref_segmask)

        edit_mask, _ = self.process_mask(input_mask, self.input_pmask, self.input_segmask)
        ma = torch.max(input_pmask)
        input_pmask[edit_mask == 1] = ma + 1
        return reference_mask, input_pmask, input_segmask, edit_mask, ma

    
    def edit(self, input_mask, ref_mask, whole_ref=False, inter=1):
        input_img = (self.input_img/127.5 - 1)
        input_img =  torch.from_numpy(input_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        reference_img = (self.ref_img/127.5 - 1)
        reference_img =  torch.from_numpy(reference_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        reference_mask, input_pmask, input_segmask, region_mask, ma = self.edit_app(input_mask, ref_mask, whole_ref)     

        input_pmask = input_pmask.float().cuda().unsqueeze(0).unsqueeze(1)
        _, mean_feat_inpt, one_hot_inpt, empty_mask_flag_inpt = model.get_appearance(input_img, input_pmask, return_all=True)

        reference_mask = reference_mask.float().cuda().unsqueeze(0).unsqueeze(1)
        _, mean_feat_ref, _, _ = model.get_appearance(reference_img, reference_mask, return_all=True)

        if mean_feat_ref.shape[1] > 1:
            mean_feat_inpt[:, ma + 1] = (1 - inter) * mean_feat_inpt[:, ma + 1] + inter*mean_feat_ref[:, 1]

        splatted_feat = torch.einsum('nmc, nmhw->nchw', mean_feat_inpt, one_hot_inpt)
        appearance = torch.nn.functional.normalize(splatted_feat) #l2 normaliz
        input_segmask =  ((input_segmask+1)/ 127.5 - 1.0).cuda().unsqueeze(0).unsqueeze(1)
        structure = torch.nn.functional.interpolate(input_segmask, (self.H, self.W))
        appearance = torch.nn.functional.interpolate(appearance, (self.H, self.W))


        return structure, appearance, region_mask, input_img
    
    def process(self, input_mask, ref_mask, prompt, a_prompt, n_prompt, 
                num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, masking=True,whole_ref=False,inter=1):
        structure, appearance, mask, img = self.edit(input_mask, ref_mask,
                                                     whole_ref=whole_ref, inter=inter)

        null_structure = torch.zeros(structure.shape).cuda() - 1
        null_appearance = torch.zeros(appearance.shape).cuda()

        null_control = torch.cat([null_structure, null_appearance], dim=1)
        structure_control = torch.cat([structure, null_appearance], dim=1)
        full_control = torch.cat([structure, appearance], dim=1)

        null_control = torch.cat([null_control for _ in range(num_samples)], dim=0)
        structure_control = torch.cat([structure_control for _ in range(num_samples)], dim=0)
        full_control = torch.cat([full_control for _ in range(num_samples)], dim=0)

        #Masking for local edit
        if not masking:
            mask, x0 = None, None
        else:
            x0 = model.encode_first_stage(img)
            x0 = x0.sample() if isinstance(x0, DiagonalGaussianDistribution) else x0 # todo: check if we can set random number
            x0 = x0 * model.scale_factor
            mask = 1 - torch.tensor(mask).unsqueeze(0).unsqueeze(1).cuda()
            mask = torch.nn.functional.interpolate(mask, x0.shape[2:]).float()
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        scale = [scale_s, scale_f, scale_t]
        print(scale)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)
        # uc_cross = model.get_unconditional_conditioning(num_samples)
        uc_cross = model.get_learned_conditioning([n_prompt] * num_samples)
        cond = {"c_concat": [full_control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt]  * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [null_control], "c_crossattn": [uc_cross]}
        un_cond_struct = {"c_concat": None if guess_mode else [structure_control], "c_crossattn": [uc_cross]}
        un_cond_struct_app = {"c_concat": None if guess_mode else [full_control], "c_crossattn": [uc_cross]}

        shape = (4, self.H // 8, self.W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale, mask=mask, x0=x0,
                                                     unconditional_conditioning=[un_cond, un_cond_struct, un_cond_struct_app ])

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = (model.decode_first_stage(samples) + 1) * 127.5
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        return [] + results    


css = """
     h1 {
  text-align: center;
}
.container {
  display: flex;  
  justify-content: space-between
}

img {
  max-width: 100%
  padding-right: 100px;
}

.image {
  flex-basis: 40%

}

.text {
  font-size: 15px;
  padding-right: 20px;
  padding-left: 0px;
}
"""

def create_app_demo():

    with gr.Row():
        gr.Markdown("## Object Level Appearance Editing")
    with gr.Row():
        gr.HTML(
            """
            <div class="container">
                <div class="text">
                    <h4> Instructions </h4>
                    <ol>
                        <li>Upload an Input Image.</li>
                        <li>Mark one of segmented objects in the <i>Select Object to Edit</i> tab.</li>
                        <li>Upload an Reference Image.</li>
                        <li>Mark  one of segmented objects in the <i>Select Reference Object</i> tab, for the reference appearance.</li>
                        <li>Enter a prompt and press <i>Run</i> button. (A very simple would also work) </li>
                    </ol>
                </div>
                <div class="image">
                    <img src="file/assets/GIF.gif" width="400"">
                </div>
            </div>
            """)
    with gr.Column():
        with gr.Row():
            img_edit = ImageComp('edit_app')
            with gr.Column():
                btn1 = gr.Button("Input Image")
                input_image = gr.Image(source='upload', label='Input Image', type="numpy",)
            with gr.Column():
                btn2 = gr.Button("Select Object to Edit")
                input_mask = gr.Image(source="upload",  label='Select Object in Input Image', type="numpy", tool="sketch")
            input_image.change(fn=img_edit.init_input_canvas, inputs=[input_image], outputs=[input_mask],  queue=False)
            
        # with gr.Row():
            with gr.Column():
                btn3 = gr.Button("Reference Image")
                ref_img = gr.Image(source='upload', label='Reference Image', type="numpy")
            with gr.Column():
                btn4 = gr.Button("Select Reference Object")
                reference_mask = gr.Image(source="upload",  label='Select Object in Refernce Image', type="numpy", tool="sketch")

            ref_img.change(fn=img_edit.init_ref_canvas, inputs=[ref_img], outputs=[reference_mask],  queue=False)
    
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value='A picture of truck')
            with gr.Column():
                interpolation = gr.Slider(label="Mixing ratio of appearance from reference object", minimum=0.1, maximum=1, value=1.0, step=0.1)
                whole_ref = gr.Checkbox(label='Use whole reference Image for appearance (Only useful for style transfers)', value=False)
        with gr.Row():
            run_button = gr.Button(label="Run")
        
        with gr.Row():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        
        with gr.Accordion("Advanced options", open=False):
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            image_resolution = gr.Slider(label="Image Resolution", minimum=512, maximum=512, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            scale_t = gr.Slider(label="Guidance Scale Text", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            scale_f = gr.Slider(label="Guidance Scale Appearance", minimum=0.1, maximum=30.0, value=8.0, step=0.1)
            scale_s = gr.Slider(label="Guidance Scale Structure", minimum=0.1, maximum=30.0, value=5.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            eta = gr.Number(label="eta (DDIM)", value=0.0)
            masking = gr.Checkbox(label='Only edit the local region', value=True)
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            
        with gr.Column():
            gr.Examples(
                examples=[['A picture of a truck', 'assets/truck.png','assets/truck2.jpeg', 892905419, 9, 7.6, 4.3],
                          ['A picture of a ironman', 'assets/ironman.webp','assets/hulk.jpeg', 709736989, 9, 7.7, 8.1],
                          ['A person skiing', 'assets/ski.jpg','assets/lava.jpg', 917723061, 9, 7.5, 4.4]],
                inputs=[prompt, input_image, ref_img, seed, scale_t, scale_f, scale_s],
                outputs=None,
                fn=None,
                cache_examples=False,
            )
    ips = [input_mask, reference_mask, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, masking, whole_ref, interpolation]
    run_button.click(fn=img_edit.process, inputs=ips, outputs=[result_gallery])



def create_struct_demo():
    with gr.Row():
        gr.Markdown("## Edit Structure (Comming soon!)")    

def create_both_demo():
    with gr.Row():
        gr.Markdown("## Edit Structure and Appearance Together (Comming soon!)")    


    
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                PAIR Diffusion
            </h1>
            <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem">
            <a href="https://vidit98.github.io/" style="color:blue;">Vidit Goel</a><sup>1*</sup>, 
            <a href="https://helia95.github.io/" style="color:blue;">Elia Peruzzo</a><sup>1,2*</sup>,
            <a href="https://yifanjiang19.github.io/" style="color:blue;">Yifan Jiang</a><sup>3</sup>, 
            <a href="https://ir1d.github.io/" style="color:blue;">Dejia Xu</a><sup>3</sup>, 
            <a href="http://disi.unitn.it/~sebe/" style="color:blue;">Nicu Sebe</a><sup>2</sup>, <br> 
            <a href=" https://people.eecs.berkeley.edu/~trevor/" style="color:blue;">Trevor Darrell</a><sup>4</sup>, 
            <a href="https://vita-group.github.io/" style="color:blue;">Zhangyang Wang</a><sup>1,3</sup>
            and <a href="https://www.humphreyshi.com/home" style="color:blue;">Humphrey Shi</a> <sup>1,5,6</sup> <br>
            [<a href="https://arxiv.org/abs/2303.17546" style="color:red;">arXiv</a>] 
            [<a href="https://github.com/Picsart-AI-Research/PAIR-Diffusion" style="color:red;">GitHub</a>]
            </h2>
            <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem">
            <sup>1</sup>Picsart AI Resarch (PAIR), <sup>2</sup>UTrenton,  <sup>3</sup>UT Austin, <sup>4</sup>UC Berkeley, <sup>5</sup>UOregon, <sup>6</sup>UIUC
            </h3>
            <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.8rem">
            We built Structure and Appearance Paired (PAIR) Diffusion  that allows reference image-guided appearance manipulation and
            structure editing of an image at an object level. PAIR diffusion models an image as composition of multiple objects and enables control 
            over structure and appearance properties of the object. Describing object appearances using text can be challenging and ambiguous, PAIR Diffusion 
            enables a user to control the appearance of an object using images. User can further use text as another degree of control for appearance. 
            Having fine-grained control over appearance and structure at object level can be beneficial for future works in video and 3D beside image editing, 
            where we need to have consistent appearance across time in case of video or across various viewing positions in case of 3D.
            </h2>
            
            </div>
            """)
    
    with gr.Tab('Edit Appearance'):
        create_app_demo()
    with gr.Tab('Edit Structure'):
        create_struct_demo()
    with gr.Tab('Edit Both'):
        create_both_demo()



block.launch(share=True)