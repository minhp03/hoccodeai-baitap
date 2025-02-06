import gradio as gr
from diffusers import DiffusionPipeline
import torch

def load_pipeline(model_name):
    model_map = {
        "v1.5": "sd-legacy/stable-diffusion-v1-5",
        "v1.0": "sd-legacy/stable-diffusion-v1-0"
    }
    pipeline = DiffusionPipeline.from_pretrained(model_map[model_name], use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    return pipeline

def generate_image(model_name,prompt,negative_prompt,seed, num_inference_steps,guidance_scale
                   ):
    pipeline = load_pipeline(model_name)
    generate_images = pipeline(
        prompt=prompt,
        model_name=model_name,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        #guidance is use to control the style of the image
        guidance_scale=guidance_scale,
        #num_inference_steps is the number of steps to generate the image ,
        seed = seed,        #more inference step lead to higher quality but increasing computation time
        num_inference_steps=num_inference_steps)
    return generate_images.images[0]
    

with gr.Blocks() as demo:
    gr.Markdown("## Diffusion picture generate")
    #1 row
    with gr.Row():
        #2 column 
        #2 place for put text , 1 button
        with gr.Column():
            model_name = gr.Dropdown(["v1.5","v1.0"],value="v1.5",info="choose your model")
            #prompt in side
            prompt = gr.Textbox(label="Prompt", placeholder="Prompt here to make a picture")
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Negative Prompt here",
                                         value="ugly, deformed, low quality")
            seed = gr.Number(label="Seed", value=0, minimum=0, maximum=50)
            num_inference_steps = gr.Number(label="Inference Steps", value=30, minimum=1, maximum=31)
            guidance_scale = gr.Number(label="Guidance Scale", value=7.5, minimum=0, maximum=10)
            generate_button = gr.Button()
        with gr.Column():
            image_output = gr.Image(label="Generated Image",height=512, width=512)
    generate_button.click(generate_image,inputs=[model_name,prompt,negative_prompt,seed,
                                                 num_inference_steps,guidance_scale
                                                 ],outputs=image_output)

demo.launch()