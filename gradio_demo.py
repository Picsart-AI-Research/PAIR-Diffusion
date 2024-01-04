import gradio as gr
from pair_diff_demo import ImageComp

# torch.cuda.set_per_process_memory_fraction(0.6)

def init_input_canvas_wrapper(obj, *args):
    return obj.init_input_canvas(*args)

def init_ref_canvas_wrapper(obj, *args):
    return obj.init_ref_canvas(*args)

def select_input_object_wrapper(obj, evt: gr.SelectData):
    return obj.select_input_object(evt)

def select_ref_object_wrapper(obj, evt: gr.SelectData):
    return obj.select_ref_object(evt)

def process_wrapper(obj, *args):
    return obj.process(*args)

def set_multi_modal_wrapper(obj, *args):
    return obj.set_multi_modal(*args)

def save_result_wrapper(obj, *args):
    return obj.save_result(*args)

def return_input_img_wrapper(obj):
    return obj.return_input_img()

def get_caption_wrapper(obj, *args):
    return obj.get_caption(*args)

def multimodal_params(b):
    if b:
        return 10, 3, 6
    else:
        return 6, 8, 9

theme = gr.themes.Soft(
    primary_hue="purple",
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", 'monospace'],
).set(
    block_label_background_fill_dark='*neutral_800'
)


css = """
    #customized_imbox {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"] {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>div {
        min-height: 450px;
    }
    #customized_imbox>div[data-testid="image"]>iframe {
        min-height: 450px;
    }
    #customized_imbox>div.unpadded_box {
        min-height: 450px;
    }
    #myinst {
        font-size: 0.8rem; 
        margin: 0rem;
        color: #6B7280;
    }
    #maskinst {
        text-align: justify;
        min-width: 1200px;
    }
    #maskinst>img {
        min-width:399px;
        max-width:450px;
        vertical-align: top;
        display: inline-block;
    }
    #maskinst:after {
        content: "";
        width: 100%;
        display: inline-block;
    }
"""

def create_app_demo():

    with gr.Row():
        gr.Markdown("## Object Level Appearance Editing")
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: left; max-width: 1200px;">
            <h3 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.8rem">
            Instructions </h3>
            <ol>
                <li>Upload an Input Image.</li>
                <li>Mark one of segmented objects in the <i>Select Object to Edit</i> tab.</li>
                <li>Upload an Reference Image.</li>
                <li>Mark  one of segmented objects in the <i>Select Reference Object</i> tab, whose appearance needs to used in the selected input object.</li>
                <li>Enter a prompt and press <i>Run</i> button. (A very simple would also work) </li>
            </ol>
            </ol>
            </div>""")
    with gr.Column():
        with gr.Row():
            img_edit = gr.State(ImageComp('edit_app'))
            with gr.Column():
                input_image = gr.Image(source='upload', label='Input Image', type="numpy",)
            with gr.Column():
                input_mask = gr.Image(source="upload",  label='Select Object in Input Image', type="numpy",)
            
            with gr.Column():
                ref_img = gr.Image(source='upload', label='Reference Image', type="numpy")
            with gr.Column():
                reference_mask = gr.Image(source="upload",  label='Select Object in Refernce Image', type="numpy")
    
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value='A picture of truck')
                mulitmod = gr.Checkbox(label='Multi-Modal', value=False)

            mulitmod.change(fn=set_multi_modal_wrapper, inputs=[img_edit, mulitmod])

            input_image.change(fn=init_input_canvas_wrapper, inputs=[img_edit, input_image],  outputs=[input_image], show_progress=True)
            input_image.select(fn=select_input_object_wrapper, inputs=[img_edit], outputs=[input_mask, prompt])

            ref_img.change(fn=init_ref_canvas_wrapper, inputs=[img_edit, ref_img],  outputs=[ref_img], show_progress=True)
            ref_img.select(fn=select_ref_object_wrapper, inputs=[img_edit], outputs=[reference_mask])

            with gr.Column():
                interpolation = gr.Slider(label="Mixing ratio of appearance from reference object", minimum=0.1, maximum=1, value=1.0, step=0.1)
                whole_ref = gr.Checkbox(label='Use whole reference Image for appearance (Only useful for style transfers)', visible=False)
            
            # clear_button.click(fn=img_edit.clear_points, inputs=[], outputs=[input_mask, reference_mask])

        with gr.Row():
            run_button = gr.Button(label="Run")
            save_button = gr.Button("Save")
        
        with gr.Row():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        
        with gr.Accordion("Advanced options", open=False):
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=1)
            image_resolution = gr.Slider(label="Image Resolution", minimum=512, maximum=512, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            scale_t = gr.Slider(label="Guidance Scale Text", minimum=0., maximum=30.0, value=6.0, step=0.1)
            scale_f = gr.Slider(label="Guidance Scale Appearance", minimum=0., maximum=30.0, value=8.0, step=0.1)
            scale_s = gr.Slider(label="Guidance Scale Structure", minimum=0., maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            eta = gr.Number(label="eta (DDIM)", value=0.0)
            masking = gr.Checkbox(label='Only edit the local region', value=True)
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            dil = gr.Slider(label="Merging region around Edge", minimum=0, maximum=0, value=0, step=0)
        
        with gr.Column():
            gr.Examples(
                examples=[['assets/examples/car.jpeg','assets/examples/ian.jpeg', '', 709736989, 6, 8, 9],
                          ['assets/examples/ian.jpeg','assets/examples/car.jpeg', '', 709736989, 6, 8, 9],
                          ['assets/examples/car.jpeg','assets/examples/ran.webp', '', 709736989, 6, 8, 9],
                          ['assets/examples/car.jpeg','assets/examples/car1.webp', '', 709736989, 6, 8, 9],
                          ['assets/examples/car1.webp','assets/examples/car.jpeg', '', 709736989, 6, 8, 9],
                           ['assets/examples/chair.jpeg','assets/examples/chair1.jpeg', '', 1106204668, 6, 8, 9],
                           ['assets/examples/house.jpeg','assets/examples/house2.jpeg', '', 1106204668, 6, 8, 9],
                           ['assets/examples/house2.jpeg','assets/examples/house.jpeg', '', 1106204668, 6, 8, 9],
                           ['assets/examples/park.webp','assets/examples/grasslands-national-park.jpeg', '', 1106204668, 6, 8, 9],
                           ['assets/examples/door.jpeg','assets/examples/door2.jpeg', '', 709736989, 6, 8, 9]],
                inputs=[input_image, ref_img, prompt, seed, scale_t, scale_f, scale_s],
                cache_examples=False,
            )

        mulitmod.change(fn=multimodal_params, inputs=[mulitmod], outputs=[scale_t, scale_f, scale_s])

    ips = [input_mask, reference_mask, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, dil, masking, whole_ref, interpolation]
    ips_save = [input_mask, prompt, a_prompt, n_prompt, ddim_steps,
                scale_s, scale_f, scale_t, seed, dil, interpolation]
    run_button.click(fn=process_wrapper, inputs=[img_edit, *ips], outputs=[result_gallery])
    save_button.click(fn=save_result_wrapper, inputs=[img_edit, *ips_save])


def create_add_obj_demo():
    with gr.Row():
        gr.Markdown("## Add Objects to Image")
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: left; max-width: 1200px;">
            <h3 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.8rem">
            Instructions </h3>
            <ol>
                <li> Upload an Input Image.</li>
                <li>Draw the precise shape of object in the image where you want to add object in <i>Draw Object</i> tab.</li>
                <li>Upload an Reference Image.</li>
                <li>Click on the object in the Reference Image tab that you want to add in the Input Image.</li>
                <li>Enter a prompt and press <i>Run</i> button. (A very simple would also work) </li>
            </ol>
            </ol>
            </div>""")
    with gr.Column():
        with gr.Row():
            img_edit = gr.State(ImageComp('add_obj'))
            with gr.Column():
                input_image = gr.Image(source='upload', label='Input Image', type="numpy",)
            with gr.Column():
                input_mask = gr.Image(source="upload",  label='Draw the desired Object', type="numpy",  tool="sketch")

            input_image.change(fn=init_input_canvas_wrapper, inputs=[img_edit, input_image],  outputs=[input_image])
            input_image.change(fn=return_input_img_wrapper, inputs=[img_edit], outputs=[input_mask],  queue=False)
            
            with gr.Column():
                ref_img = gr.Image(source='upload', label='Reference Image', type="numpy")
            with gr.Column():
                reference_mask = gr.Image(source="upload",  label='Selected Object in Refernce Image', type="numpy")

            ref_img.change(fn=init_ref_canvas_wrapper, inputs=[img_edit, ref_img], outputs=[ref_img],  queue=False)
            # ref_img.upload(fn=img_edit.init_ref_canvas, inputs=[ref_img],  outputs=[ref_img])
            ref_img.select(fn=select_ref_object_wrapper, inputs=[img_edit], outputs=[reference_mask])

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value='A picture of truck')
            mulitmod = gr.Checkbox(label='Multi-Modal', value=False, visible=False)

        mulitmod.change(fn=set_multi_modal_wrapper, inputs=[img_edit, mulitmod])

        with gr.Row():
            run_button = gr.Button(label="Run")
            save_button = gr.Button("Save")
        
        with gr.Row():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        
        with gr.Accordion("Advanced options", open=False):
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=1)
            # image_resolution = gr.Slider(label="Image Resolution", minimum=512, maximum=512, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            dil = gr.Slider(label="Merging region around Edge", minimum=0, maximum=5, value=2, step=1)
            scale_t = gr.Slider(label="Guidance Scale Text", minimum=0., maximum=30.0, value=6.0, step=0.1)
            scale_f = gr.Slider(label="Guidance Scale Appearance", minimum=0., maximum=30.0, value=8.0, step=0.1)
            scale_s = gr.Slider(label="Guidance Scale Structure", minimum=0., maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            eta = gr.Number(label="eta (DDIM)", value=0.0)
            masking = gr.Checkbox(label='Only edit the local region', value=True)
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        
        mulitmod.change(fn=multimodal_params, inputs=[mulitmod], outputs=[scale_t, scale_f, scale_s])

        with gr.Column():
            gr.Examples(
                examples=[['assets/examples/chair.jpeg','assets/examples/carpet2.webp', 'A picture of  living room with carpet', 892905419, 6, 8, 9],
                          ['assets/examples/chair.jpeg','assets/examples/chair1.jpeg', 'A picture of  living room with a orange and white sofa', 892905419, 6, 8, 9],
                          ['assets/examples/park.webp','assets/examples/dog.jpeg', 'A picture of  dog in the park', 892905419, 6, 8, 9]],
                inputs=[input_image, ref_img, prompt, seed, scale_t, scale_f, scale_s],
                outputs=None,
                fn=None,
                cache_examples=False,
            )
    ips = [input_mask, reference_mask, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, dil, masking]
    ips_save = [input_mask, prompt, a_prompt, n_prompt, ddim_steps,
                scale_s, scale_f, scale_t, seed, dil]
    run_button.click(fn=process_wrapper, inputs=[img_edit, *ips], outputs=[result_gallery])
    save_button.click(fn=save_result_wrapper, inputs=[img_edit, *ips_save])

def create_obj_variation_demo():
    with gr.Row():
        gr.Markdown("## Objects Variation")
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: left; max-width: 1200px;">
            <h3 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.8rem">
            Instructions </h3>
            <ol>
                <li> Upload an Input Image.</li>
                <li>Click on object to have variations</li>
                <li>Press <i>Run</i> button</li>
            </ol>
            </ol>
            </div>""")

    with gr.Column():
        with gr.Row():
            img_edit = gr.State(ImageComp('edit_app'))
            with gr.Column():
                input_image = gr.Image(source='upload', label='Input Image', type="numpy",)
            with gr.Column():
                input_mask = gr.Image(source="upload",  label='Select Object in Input Image', type="numpy",)

        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value='')
            mulitmod = gr.Checkbox(label='Multi-Modal', value=False)

        
        mulitmod.change(fn=set_multi_modal_wrapper, inputs=[img_edit, mulitmod])

        input_image.change(fn=init_input_canvas_wrapper, inputs=[img_edit, input_image],  outputs=[input_image])
        input_image.select(fn=select_input_object_wrapper, inputs=[img_edit], outputs=[input_mask, prompt])
        input_image.change(fn=init_ref_canvas_wrapper, inputs=[img_edit, input_image], outputs=[],  queue=False)
        input_image.select(fn=select_ref_object_wrapper, inputs=[img_edit], outputs=[])
                
        with gr.Row():
            run_button = gr.Button(label="Run")
            save_button = gr.Button("Save")
        
        with gr.Row():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        
        with gr.Accordion("Advanced options", open=False):
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=2)
            # image_resolution = gr.Slider(label="Image Resolution", minimum=512, maximum=512, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            dil = gr.Slider(label="Merging region around Edge", minimum=0, maximum=5, value=2, step=1)
            scale_t = gr.Slider(label="Guidance Scale Text", minimum=0.0, maximum=30.0, value=6.0, step=0.1)
            scale_f = gr.Slider(label="Guidance Scale Appearance", minimum=0.0, maximum=30.0, value=8.0, step=0.1)
            scale_s = gr.Slider(label="Guidance Scale Structure", minimum=0.0, maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            eta = gr.Number(label="eta (DDIM)", value=0.0)
            masking = gr.Checkbox(label='Only edit the local region', value=True)
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        

        mulitmod.change(fn=multimodal_params, inputs=[mulitmod], outputs=[scale_t, scale_f, scale_s])

        with gr.Column():
            gr.Examples(
                examples=[['assets/examples/chair.jpeg' , 892905419, 6, 8, 9],
                          ['assets/examples/chair1.jpeg', 892905419, 6, 8, 9],
                          ['assets/examples/park.webp',  892905419, 6, 8, 9],
                          ['assets/examples/car.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/ian.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/chair.jpeg', 1106204668, 6, 8, 9],
                          ['assets/examples/door.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/carpet2.webp',  892905419, 6, 8, 9],
                          ['assets/examples/house.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/house2.jpeg', 709736989, 6, 8, 9],],
                inputs=[input_image, seed, scale_t, scale_f, scale_s],
                outputs=None,
                fn=None,
                cache_examples=False,
            )
    ips = [input_mask, input_mask, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, dil, masking]
    ips_save = [input_mask, prompt, a_prompt, n_prompt, ddim_steps,
                scale_s, scale_f, scale_t, seed, dil]
    run_button.click(fn=process_wrapper, inputs=[img_edit, *ips], outputs=[result_gallery])
    save_button.click(fn=save_result_wrapper, inputs=[img_edit, *ips_save])


def create_free_form_obj_variation_demo():
    with gr.Row():
        gr.Markdown("## Objects Variation")
    with gr.Row():
        gr.HTML(
            """
            <div style="text-align: left; max-width: 1200px;">
            <h3 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.8rem">
            Instructions </h3>
            <ol>
                <li> Upload an Input Image.</li>
                <li>Mask the region that you want to have variation</li>
                <li>Press <i>Run</i> button</li>
            </ol>
            </ol>
            </div>""")

    with gr.Column():
        with gr.Row():
            img_edit = gr.State(ImageComp('edit_app'))
            with gr.Column():
                input_image = gr.Image(source='upload', label='Input Image', type="numpy", )
            with gr.Column():
                input_mask = gr.Image(source="upload",  label='Select Object in Input Image', type="numpy", tool="sketch")
        
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value='')
            ignore_structure = gr.Checkbox(label='Ignore Structure (Please provide a good caption)', visible=False)
            mulitmod = gr.Checkbox(label='Multi-Modal', value=False)

        mulitmod.change(fn=set_multi_modal_wrapper, inputs=[img_edit, mulitmod])

        input_image.change(fn=init_input_canvas_wrapper, inputs=[img_edit, input_image],  outputs=[input_mask])
        input_mask.edit(fn=get_caption_wrapper, inputs=[img_edit, input_mask], outputs=[prompt])
        input_image.change(fn=init_ref_canvas_wrapper, inputs=[img_edit, input_image], outputs=[],  queue=False)
        # input_image.select(fn=select_ref_object_wrapper, inputs=[img_edit], outputs=[])
        
        # input_image.edit(fn=img_edit.vis_mask, inputs=[input_image], outputs=[input_mask])
                
        with gr.Row():
            run_button = gr.Button(label="Run")
            save_button = gr.Button("Save")
        
        with gr.Row():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=4, height='auto')
        
        with gr.Accordion("Advanced options", open=False):
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=4, step=2)
            # image_resolution = gr.Slider(label="Image Resolution", minimum=512, maximum=512, value=512, step=64)
            strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='Guess Mode', value=False)
            ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
            dil = gr.Slider(label="Merging region around Edge", minimum=0, maximum=5, value=2, step=1)
            scale_t = gr.Slider(label="Guidance Scale Text", minimum=0.0, maximum=30.0, value=6.0, step=0.1)
            scale_f = gr.Slider(label="Guidance Scale Appearance", minimum=0.0, maximum=30.0, value=8.0, step=0.1)
            scale_s = gr.Slider(label="Guidance Scale Structure", minimum=0.0, maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            eta = gr.Number(label="eta (DDIM)", value=0.0)
            masking = gr.Checkbox(label='Only edit the local region', value=True)
            free_form_obj_var = gr.Checkbox(label='', value=True)
            a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="Negative Prompt",
                                    value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
            interpolation = gr.Slider(label="Mixing ratio of appearance from reference object", minimum=0.0, maximum=0.1, step=0.1)
        
        mulitmod.change(fn=multimodal_params, inputs=[mulitmod], outputs=[scale_t, scale_f, scale_s])

        with gr.Column():
            gr.Examples(
                examples=[['assets/examples/chair.jpeg' , 892905419, 6, 8, 9],
                          ['assets/examples/chair1.jpeg', 892905419, 6, 8, 9],
                          ['assets/examples/park.webp',  892905419, 6, 8, 9],
                          ['assets/examples/car.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/ian.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/chair.jpeg', 1106204668, 6, 8, 9],
                          ['assets/examples/door.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/carpet2.webp',  892905419, 6, 8, 9],
                          ['assets/examples/house.jpeg',  709736989, 6, 8, 9],
                          ['assets/examples/house2.jpeg', 709736989, 6, 8, 9],],
                inputs=[input_image, seed, scale_t, scale_f, scale_s],
                outputs=None,
                fn=None,
                cache_examples=False,
            )
    ips = [input_mask, input_mask, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, 
                scale_s, scale_f, scale_t, seed, eta, dil, masking, free_form_obj_var, dil, free_form_obj_var, ignore_structure]
    ips_save = [input_mask, prompt, a_prompt, n_prompt, ddim_steps,
                scale_s, scale_f, scale_t, seed, dil, interpolation, free_form_obj_var]
    run_button.click(fn=process_wrapper, inputs=[img_edit, *ips], outputs=[result_gallery])
    save_button.click(fn=save_result_wrapper, inputs=[img_edit, *ips_save])

    
block = gr.Blocks(css=css, theme=theme).queue()
with block:
    gr.HTML(
            """
            <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
            <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">
                PAIR Diffusion: A Comprehensive Multimodal Object-Level Image Editor
            </h1>
            <h3 style="margin-top: 0.6rem; margin-bottom: 1rem">Picsart AI Research</h3>
            <h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.8rem; margin-bottom: 0.8rem">
            PAIR diffusion provides comprehensive multi-modal editing capabilities to edit real images without the need of inverting them. The current suite contains
         <span style="color: #01feee;">Object Variation</span>, <span style="color: #4f82d9;">Edit Appearance of any object using a reference image and text</span>, 
            <span style="color: #d402bf;">Add any object from a reference image in the input image</span>. This operations can be mixed with each other to 
            develop new editing operations in future.
            </ul>
            </h2>
            </div>
            """)
    
    with gr.Tab('Edit Appearance'):
        create_app_demo()
    with gr.Tab('Object Variation Free Form Mask'):
        create_free_form_obj_variation_demo()
    with gr.Tab('Object Variation'):
        create_obj_variation_demo()
    with gr.Tab('Add Objects'):
        create_add_obj_demo()

block.queue(max_size=20)
block.launch(share=True)
            
