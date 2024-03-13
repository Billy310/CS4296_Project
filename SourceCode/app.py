from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    DiffusionPipeline,
)
import torch
import random
from flask import Flask, Response, request, jsonify, render_template, send_file
from flask_cors import CORS
import io
from PIL import Image
import base64
from flask_socketio import SocketIO
from asyncio import sleep
from user import User

# socketid: str
progress_percentage_user = 0.0
total_step_gen = 40
finished = False
app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)

# model_id = "stabilityai/stable-diffusion-2-1"
# model_id = "WarriorMama777/AbyssOrangeMix2"

model_id = "./AnythingXL_v50/AnythingXL_v50.safetensors"


# pipe = StableDiffusionPipeline.from_pretrained(
#     model_id, torch_dtype=torch.float16, use_safetensors=False
# )
pipe = StableDiffusionPipeline.from_single_file(model_id)


# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.unet.load_attn_procs("./model/lora/anpan.safetensors")
pipe.load_lora_weights("./model/lora/", weight_name="J_illustration.safetensors")
pipe = pipe.to("cuda")

# pipe.to("cuda")
# pipe.load_lora_weights(".", weight_name="./model/lora/anpan.safetensors")
# pipe.lora_state_dict(pretrained_model_name_or_path_or_dict="./model/lora/anpan.safetensors")

# pipe.safety_checker = lambda images, clip_input: (images, False)


# pipe.enable_attention_slicing()


def generate_image(prompt):

    image = pipe(
        prompt,
        negative_prompt = "easy_negative, NSFW",
        guidance_scale = 7,
        # negative_prompt="nsfw, (worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), (blush:1.2), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature",
        num_inference_steps=total_step_gen,
        callback=progress,
        callback_steps=1,
    ).images[0]

    global finished
    finished = True
    # print(image.hidden_states)
    return image


@app.route("/generate", methods=["POST"])
def generate():
    global finished
    finished = False
    global progress_percentage_user
    progress_percentage_user = 0.0

    socketid = request.args.get("socketid")
    animal = request.values.get("animal");
    style_ = request.values.get("style_");
    action_ = request.values.get("action_");

    prompt = "master piece, high quality"+", a "+style_+" "+animal+","+action_+"<lora:J_illustration:0.8> j_illustration"
    print("prompt:", prompt)
    image = generate_image(prompt)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # finished = True

    return send_file(
        io.BytesIO(image_bytes),
        mimetype="image/jpeg",
        download_name="img.jpg",
        as_attachment=True,
    )


@app.route("/img_to_img", methods=["POST"])
def imgtoimg():

    global progress_percentage_user
    progress_percentage_user = 0.0
    global finished
    finished = False
    image = generate_image("animate")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    finished = True

    return send_file(
        io.BytesIO(image_bytes),
        mimetype="image/jpeg",
        download_name="img.jpg",
        as_attachment=True,
    )

    # init_image = Image.open()


def progress(step, timestep, latents):
    # print(step,timestep,latents[0][0][0][0])

    global progress_percentage_user
    progress_percentage_user = float((step / total_step_gen))
    # progress_percentage1 = progress_percentage
    # print(f"Progress: {progress_percentage_user}%")


@app.route("/progressInfo/<socketid>", methods=["POST"])
async def progressInfo(socketid):

    while not finished:

        print(progress_percentage_user)
        socketio.emit("update progress", progress_percentage_user, to=socketid)
        await sleep(0.1)

    socketio.emit("update progress", 1, to=socketid)
    # finished = False;
    return Response(status=204)


@app.route("/")
def index():
    return render_template("index.html")


# Start an http server and expose an api endpoint that takes in a prompt and returns an image.
def main():
    print("Starting server...")
    # app.run(host="localhost", port=5000)
    socketio.run(app, host="localhost", port=5000)


if __name__ == "__main__":
    main()

# device = "cuda"
# model_id_or_path = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
# pipe = pipe.to(device)

# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

# response = requests.get(url)
# init_image = Image.open(BytesIO(response.content)).convert("RGB")
# init_image = init_image.resize((768, 512))

# prompt = "A fantasy landscape, trending on artstation"

# images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images
# images[0].save("fantasy_landscape.png")
