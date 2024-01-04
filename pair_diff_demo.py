import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import json
import datetime
from huggingface_hub import hf_hub_url, hf_hub_download

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.OneFormer import OneformerSegmenter
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSamplerSpaCFG
from ldm.models.autoencoder import DiagonalGaussianDistribution


SEGMENT_MODEL_DICT = {
    'Oneformer': OneformerSegmenter,
}

MASK_MODEL_DICT = {
    'Oneformer': OneformerSegmenter,    
}

urls = {
    'shi-labs/oneformer_coco_swin_large': ['150_16_swin_l_oneformer_coco_100ep.pth'],
    'PAIR/PAIR-diffusion-sdv15-coco-finetune': ['model_e91.ckpt']
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


#main model
model = create_model('configs/pair_diff.yaml').cpu()
model.load_state_dict(load_state_dict(WTS_DICT['PAIR/PAIR-diffusion-sdv15-coco-finetune'], location='cuda'))

save_dir = 'results/'

model = model.cuda()
ddim_sampler = DDIMSamplerSpaCFG(model)
save_memory = False


class ImageComp:
    def __init__(self, edit_operation):
        self.input_img = None
        self.input_pmask = None
        self.input_segmask = None
        self.input_mask = None
        self.input_points = []
        self.input_scale = 1

        self.ref_img = None
        self.ref_pmask = None
        self.ref_segmask = None
        self.ref_mask = None
        self.ref_points = []
        self.ref_scale = 1

        self.multi_modal = False

        self.H = None
        self.W = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.edit_operation = edit_operation
        self.init_segmentation_model()
        os.makedirs(save_dir, exist_ok=True)

        self.base_prompt = 'A picture of {}'

    def init_segmentation_model(self, mask_model='Oneformer', segment_model='Oneformer'):
        self.segment_model_name = segment_model
        self.mask_model_name = mask_model

        self.segment_model = SEGMENT_MODEL_DICT[segment_model](WTS_DICT['shi-labs/oneformer_coco_swin_large'])

        if mask_model == 'Oneformer' and segment_model == 'Oneformer':
            self.mask_model_inp = self.segment_model
            self.mask_model_ref = self.segment_model
        else:
            self.mask_model_inp = MASK_MODEL_DICT[mask_model]()
            self.mask_model_ref = MASK_MODEL_DICT[mask_model]()

        print(f"Segmentation Models initialized with {mask_model} as mask and {segment_model} as segment")

    def init_input_canvas(self, img):

        img =  HWC3(img)
        img = resize_image(img, 512)
        if self.segment_model_name == 'Oneformer':
            detected_seg = self.segment_model(img, 'semantic')
        elif self.segment_model_name == 'SAM':
            raise NotImplementedError
        
        if self.mask_model_name == 'Oneformer':
            detected_mask = self.mask_model_inp(img, 'panoptic')[0]
        elif self.mask_model_name == 'SAM':
            detected_mask = self.mask_model_inp(img)
        
        self.input_points = []
        self.input_img = img
        self.input_pmask = detected_mask
        self.input_segmask = detected_seg
        self.H = img.shape[0]
        self.W = img.shape[1]

        return img
    
    def init_ref_canvas(self, img):

        img =  HWC3(img)
        img = resize_image(img, 512)
        if self.segment_model_name == 'Oneformer':
            detected_seg = self.segment_model(img, 'semantic')
        elif self.segment_model_name == 'SAM':
            raise NotImplementedError
        
        if self.mask_model_name == 'Oneformer':
            detected_mask = self.mask_model_ref(img, 'panoptic')[0]
        elif self.mask_model_name == 'SAM':
            detected_mask = self.mask_model_ref(img)

        self.ref_points = []
        print("Initialized ref", img.shape)
        self.ref_img = img
        self.ref_pmask = detected_mask
        self.ref_segmask = detected_seg

        return img

    def select_input_object(self, evt: gr.SelectData):
        idx = list(np.array(evt.index) * self.input_scale)
        self.input_points.append(idx)
        if self.mask_model_name == 'Oneformer':
            mask = self._get_mask_from_panoptic(np.array(self.input_points), self.input_pmask)
        else:
            mask = self.mask_model_inp(self.input_img, self.input_points)

        c_ids = self.input_segmask[np.array(self.input_points)[:,1], np.array(self.input_points)[:,0]]
        unique_ids, counts = torch.unique(c_ids, return_counts=True)
        c_id = int(unique_ids[torch.argmax(counts)].cpu().detach().numpy())
        category = self.segment_model.metadata.stuff_classes[c_id]
        # print(self.segment_model.metadata.stuff_classes)

        self.input_mask = mask
        mask = mask.cpu().numpy()
        output = mask[:,:,None] * self.input_img + (1 - mask[:,:,None]) * self.input_img * 0.2
        return output.astype(np.uint8), self.base_prompt.format(category)

    def select_ref_object(self, evt: gr.SelectData):
        idx = list(np.array(evt.index) * self.ref_scale)
        self.ref_points.append(idx)
        if self.mask_model_name == 'Oneformer':
            mask = self._get_mask_from_panoptic(np.array(self.ref_points), self.ref_pmask)
        else:
            mask = self.mask_model_ref(self.ref_img, self.ref_points)
        c_ids = self.ref_segmask[np.array(self.ref_points)[:,1], np.array(self.ref_points)[:,0]]
        unique_ids, counts = torch.unique(c_ids, return_counts=True)
        c_id = int(unique_ids[torch.argmax(counts)].cpu().detach().numpy())
        category = self.segment_model.metadata.stuff_classes[c_id]
        print("Category of reference object is:", category)

        self.ref_mask = mask
        mask = mask.cpu().numpy()
        output = mask[:,:,None] * self.ref_img + (1 - mask[:,:,None]) * self.ref_img * 0.2
        return output.astype(np.uint8)
    
    def clear_points(self):
        self.input_points = []
        self.ref_points = []
        zeros_inp = np.zeros(self.input_img.shape)
        zeros_ref = np.zeros(self.ref_img.shape)
        return zeros_inp, zeros_ref
    
    def return_input_img(self):
        return self.input_img


    def _get_mask_from_panoptic(self, points, panoptic_mask):
        panoptic_mask_ = panoptic_mask + 1
        ids = panoptic_mask_[points[:,1], points[:,0]]
        unique_ids, counts = torch.unique(ids, return_counts=True)
        mask_id = unique_ids[torch.argmax(counts)]
        final_mask = torch.zeros(panoptic_mask.shape).cuda()
        final_mask[panoptic_mask_ == mask_id] = 1

        return final_mask


    def _process_mask(self, mask, panoptic_mask, segmask):
        obj_class = mask * (segmask + 1)
        unique_ids, counts = torch.unique(obj_class, return_counts=True)
        obj_class = unique_ids[torch.argmax(counts[1:]) + 1] - 1
        return mask, obj_class
    
    
    def _edit_app(self, whole_ref):
        """
            Manipulates the panoptic mask of input image to change appearance
        """
        input_pmask = self.input_pmask
        input_segmask = self.input_segmask

        if whole_ref:
            reference_mask = torch.ones(self.ref_pmask.shape).cuda()
        else:
            reference_mask, _ = self._process_mask(self.ref_mask, self.ref_pmask, self.ref_segmask)

        edit_mask, _ = self._process_mask(self.input_mask, self.input_pmask, self.input_segmask)
        # tmp = cv2.dilate(edit_mask.squeeze().cpu().numpy(), self.kernel, iterations = 2)
        # region_mask = torch.tensor(tmp).cuda()
        region_mask = edit_mask
        ma = torch.max(input_pmask)
        
        input_pmask[edit_mask == 1] = ma + 1
        return reference_mask, input_pmask, input_segmask, region_mask, ma

    def _add_object(self, input_mask, dilation_fac):
        """
        Manipulates the panooptic mask of input image for adding objects
        Args:
            input_mask (numpy array): Region where new objects needs to be added
            dilation factor (float): Controls edge merging region for adding objects

        """
        input_pmask = self.input_pmask
        input_segmask = self.input_segmask
        reference_mask, obj_class = self._process_mask(self.ref_mask, self.ref_pmask, self.ref_segmask)
        
        tmp = cv2.dilate(input_mask['mask'][:, :, 0], self.kernel, iterations = int(dilation_fac))
        region = torch.tensor(tmp)
        region_mask = torch.zeros_like(region).cuda()
        region_mask[region > 127] = 1

        mask_ = torch.tensor(input_mask['mask'][:, :, 0])
        edit_mask = torch.zeros_like(mask_).cuda()
        edit_mask[mask_ > 127] = 1
        ma = torch.max(input_pmask)
        input_pmask[edit_mask == 1] = ma + 1
        print(obj_class)
        input_segmask[edit_mask == 1] = obj_class.long()
       
        return reference_mask, input_pmask, input_segmask, region_mask, ma
    
    def _edit(self, input_mask, ref_mask, dilation_fac=1, whole_ref=False, inter=1):
        """
        Entry point for all the appearance editing and add objects operations. The function manipulates the
        appearance vectors and structure based on user input
        Args:
            input mask (numpy array): Region in input image which needs to be edited
            dilation factor (float): Controls edge merging region for adding objects
            whole_ref (bool): Flag for specifying if complete reference image should be used
            inter (float): Interpolation of appearance between the reference appearance and the input appearance.
        """
        input_img = (self.input_img/127.5 - 1)
        input_img =  torch.from_numpy(input_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        reference_img = (self.ref_img/127.5 - 1)
        reference_img =  torch.from_numpy(reference_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        if self.edit_operation == 'add_obj':
            reference_mask, input_pmask, input_segmask, region_mask, ma = self._add_object(input_mask, dilation_fac)
        elif self.edit_operation == 'edit_app':
            reference_mask, input_pmask, input_segmask, region_mask, ma = self._edit_app(whole_ref)    

        #concat featurees
        input_pmask = input_pmask.float().cuda().unsqueeze(0).unsqueeze(1)
        _, mean_feat_inpt_conc, one_hot_inpt_conc, _ = model.get_appearance(model.appearance_net_conc, model.app_layer_conc, input_img, input_pmask, return_all=True)

        reference_mask = reference_mask.float().cuda().unsqueeze(0).unsqueeze(1)
        _, mean_feat_ref_conc, _, _ = model.get_appearance(model.appearance_net_conc, model.app_layer_conc, reference_img, reference_mask, return_all=True)

        # if mean_feat_ref.shape[1] > 1:
        if isinstance(mean_feat_inpt_conc, list):
            appearance_conc = []
            for i in range(len(mean_feat_inpt_conc)):
                mean_feat_inpt_conc[i][:, ma + 1] = (1 - inter) * mean_feat_inpt_conc[i][:, ma + 1] + inter*mean_feat_ref_conc[i][:, 1]
                splatted_feat_conc = torch.einsum('nmc, nmhw->nchw', mean_feat_inpt_conc[i], one_hot_inpt_conc)
                splatted_feat_conc = torch.nn.functional.normalize(splatted_feat_conc)
                splatted_feat_conc = torch.nn.functional.interpolate(splatted_feat_conc, (self.H//8, self.W//8))
                appearance_conc.append(splatted_feat_conc)
            appearance_conc = torch.cat(appearance_conc, dim=1)
        else:
            print("manipulating")
            mean_feat_inpt_conc[:, ma + 1] = (1 - inter) * mean_feat_inpt_conc[:, ma + 1] + inter*mean_feat_ref_conc[:, 1]

            splatted_feat_conc = torch.einsum('nmc, nmhw->nchw', mean_feat_inpt_conc, one_hot_inpt_conc)
            appearance_conc = torch.nn.functional.normalize(splatted_feat_conc) #l2 normaliz
            appearance_conc = torch.nn.functional.interpolate(appearance_conc, (self.H//8, self.W//8))

        #cross attention features
        _, mean_feat_inpt_ca, one_hot_inpt_ca, _ = model.get_appearance(model.appearance_net_ca, model.app_layer_ca, input_img, input_pmask, return_all=True)

        _, mean_feat_ref_ca, _, _ = model.get_appearance(model.appearance_net_ca, model.app_layer_ca, reference_img, reference_mask, return_all=True)

        # if mean_feat_ref.shape[1] > 1:
        if isinstance(mean_feat_inpt_ca, list):
            appearance_ca = []
            for i in range(len(mean_feat_inpt_ca)):
                mean_feat_inpt_ca[i][:, ma + 1] = (1 - inter) * mean_feat_inpt_ca[i][:, ma + 1] + inter*mean_feat_ref_ca[i][:, 1]
                splatted_feat_ca = torch.einsum('nmc, nmhw->nchw', mean_feat_inpt_ca[i], one_hot_inpt_ca)
                splatted_feat_ca = torch.nn.functional.normalize(splatted_feat_ca)
                splatted_feat_ca = torch.nn.functional.interpolate(splatted_feat_ca, (self.H//8, self.W//8))
                appearance_ca.append(splatted_feat_ca)
        else:
            print("manipulating")
            mean_feat_inpt_ca[:, ma + 1] = (1 - inter) * mean_feat_inpt_ca[:, ma + 1] + inter*mean_feat_ref_ca[:, 1]

            splatted_feat_ca = torch.einsum('nmc, nmhw->nchw', mean_feat_inpt_ca, one_hot_inpt_ca)
            appearance_ca = torch.nn.functional.normalize(splatted_feat_ca) #l2 normaliz
            appearance_ca = torch.nn.functional.interpolate(appearance_ca, (self.H//8, self.W//8))



        input_segmask =  ((input_segmask+1)/ 127.5 - 1.0).cuda().unsqueeze(0).unsqueeze(1)
        structure = torch.nn.functional.interpolate(input_segmask, (self.H//8, self.W//8))
        

        return structure, appearance_conc, appearance_ca, region_mask, input_img
    
    def _edit_obj_var(self, input_mask, ignore_structure):
        input_img = (self.input_img/127.5 - 1)
        input_img =  torch.from_numpy(input_img.astype(np.float32)).cuda().unsqueeze(0).permute(0,3,1,2)

        
        input_pmask = self.input_pmask
        input_segmask = self.input_segmask

        ma = torch.max(input_pmask)
        mask_ = torch.tensor(input_mask['mask'][:, :, 0])
        edit_mask = torch.zeros_like(mask_).cuda()
        edit_mask[mask_ > 127] = 1
        tmp = edit_mask * (input_pmask + ma + 1)
        if ignore_structure:
            tmp = edit_mask

        input_pmask = tmp * edit_mask + (1 - edit_mask) * input_pmask

        input_pmask = input_pmask.float().cuda().unsqueeze(0).unsqueeze(1)

        mask_ca_feat = self.input_pmask.float().cuda().unsqueeze(0).unsqueeze(1) if ignore_structure else input_pmask
        print(torch.unique(mask_ca_feat))
        
        appearance_conc,_,_,_ = model.get_appearance(model.appearance_net_conc, model.app_layer_conc, input_img, input_pmask, return_all=True)
        appearance_ca = model.get_appearance(model.appearance_net_ca, model.app_layer_ca,  input_img, mask_ca_feat)

        appearance_conc = torch.nn.functional.interpolate(appearance_conc, (self.H//8, self.W//8))
        appearance_ca = [torch.nn.functional.interpolate(ap, (self.H//8, self.W//8)) for ap in appearance_ca]

        input_segmask =  ((input_segmask+1)/ 127.5 - 1.0).cuda().unsqueeze(0).unsqueeze(1)
        structure = torch.nn.functional.interpolate(input_segmask, (self.H//8, self.W//8))
        

        tmp = input_mask['mask'][:, :, 0]
        region = torch.tensor(tmp)
        mask = torch.zeros_like(region).cuda()
        mask[region > 127] = 1

        return structure, appearance_conc, appearance_ca, mask, input_img
    
    def get_caption(self, mask):
        """
        Generates the captions based on a set template
        Args:
            mask (numpy array): Region of image based on which caption needs to be generated
        """
        mask = mask['mask'][:, :, 0]
        region = torch.tensor(mask).cuda()
        mask = torch.zeros_like(region)
        mask[region > 127] = 1

        if torch.sum(mask) == 0:
            return ""

        c_ids = self.input_segmask * mask
        unique_ids, counts = torch.unique(c_ids, return_counts=True)
        c_id = int(unique_ids[torch.argmax(counts[1:]) + 1].cpu().detach().numpy())
        category = self.segment_model.metadata.stuff_classes[c_id]

        return self.base_prompt.format(category)
    
    def save_result(self, input_mask, prompt, a_prompt, n_prompt, 
                ddim_steps, scale_s, scale_f, scale_t, seed, dilation_fac=1,inter=1,
                  free_form_obj_var=False, ignore_structure=False):
        """
        Saves the current results with all the meta data
        """

        meta_data = {}
        meta_data['prompt'] = prompt
        meta_data['a_prompt'] = a_prompt
        meta_data['n_prompt'] = n_prompt
        meta_data['seed'] = seed
        meta_data['ddim_steps'] = ddim_steps
        meta_data['scale_s'] = scale_s
        meta_data['scale_f'] = scale_f
        meta_data['scale_t'] = scale_t
        meta_data['inter'] = inter
        meta_data['dilation_fac'] = dilation_fac
        meta_data['edit_operation'] = self.edit_operation
        
        uuid = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        os.makedirs(f'{save_dir}/{uuid}')

        with open(f'{save_dir}/{uuid}/meta.json', "w") as outfile:
                json.dump(meta_data, outfile)
        cv2.imwrite(f'{save_dir}/{uuid}/input.png', self.input_img[:,:,::-1])
        cv2.imwrite(f'{save_dir}/{uuid}/ref.png', self.ref_img[:,:,::-1])
        if self.ref_mask is not None:
            cv2.imwrite(f'{save_dir}/{uuid}/ref_mask.png', self.ref_mask.cpu().squeeze().numpy() * 200)
        for i in range(len(self.results)):
            cv2.imwrite(f'{save_dir}/{uuid}/edit{i}.png', self.results[i][:,:,::-1])

        if self.edit_operation == 'add_obj' or free_form_obj_var:
            cv2.imwrite(f'{save_dir}/{uuid}/input_mask.png', input_mask['mask'] * 200)
        else:
            cv2.imwrite(f'{save_dir}/{uuid}/input_mask.png', self.input_mask.cpu().squeeze().numpy() * 200)

        print("Saved results at", f'{save_dir}/{uuid}')
            

    def process(self, input_mask, ref_mask, prompt, a_prompt, n_prompt, 
                num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, dilation_fac=1,masking=True,whole_ref=False,inter=1,
                  free_form_obj_var=False, ignore_structure=False):

        print(prompt)
        if free_form_obj_var:
            print("Free form")
            structure, appearance_conc, appearance_ca, mask, img = self._edit_obj_var(input_mask, ignore_structure)
        else:
            structure, appearance_conc, appearance_ca, mask, img = self._edit(input_mask, ref_mask, dilation_fac=dilation_fac,
                                                     whole_ref=whole_ref, inter=inter)
        
        input_pmask =  torch.nn.functional.interpolate(self.input_pmask.cuda().unsqueeze(0).unsqueeze(1).float(), (self.H//8, self.W//8))
        input_pmask = input_pmask.to(memory_format=torch.contiguous_format)

        
        if isinstance(appearance_ca, list):
            null_appearance_ca = [torch.zeros(a.shape).cuda() for a in appearance_ca]
            null_appearance_conc = torch.zeros(appearance_conc.shape).cuda()
            null_structure = torch.zeros(structure.shape).cuda() - 1

            null_control = [torch.cat([null_structure, napp, input_pmask * 0], dim=1) for napp in null_appearance_ca]
            structure_control = [torch.cat([structure, napp, input_pmask], dim=1) for napp in null_appearance_ca]
            full_control = [torch.cat([structure, napp, input_pmask], dim=1) for napp in appearance_ca]

            null_control.append(torch.cat([null_structure, null_appearance_conc, null_structure * 0], dim=1))
            structure_control.append(torch.cat([structure, null_appearance_conc, null_structure], dim=1))
            full_control.append(torch.cat([structure, appearance_conc, input_pmask], dim=1))

            null_control = [torch.cat([nc for _ in range(num_samples)], dim=0) for nc in null_control]
            structure_control = [torch.cat([sc for _ in range(num_samples)], dim=0) for sc in structure_control]
            full_control = [torch.cat([fc for _ in range(num_samples)], dim=0) for fc in full_control]

        #Masking for local edit
        if not masking:
            mask, x0 = None, None
        else:
            x0 = model.encode_first_stage(img)
            x0 = x0.sample() if isinstance(x0, DiagonalGaussianDistribution) else x0 # todo: check if we can set random number
            x0 = x0 * model.scale_factor
            mask = 1 - torch.tensor(mask).unsqueeze(0).unsqueeze(1).cuda()
            mask = torch.nn.functional.interpolate(mask.float(), x0.shape[2:]).float()
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        scale = [scale_s, scale_f, scale_t]
        print(scale)
        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_cross = model.get_learned_conditioning([n_prompt] * num_samples)
        c_cross = model.get_learned_conditioning([prompt + ', ' + a_prompt]  * num_samples)
        cond = {"c_concat": [null_control], "c_crossattn": [c_cross]}
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
        self.results = results
        return [] + results