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

total_step_gen = 40
finished = False
app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
progress_percentage_user = {}


def generate_image(prompt, socketid):
    def progress_with_socketid(step, timestep, latents):
        progress(step, timestep, latents, socketid)
    model_id = "./AnythingXL_v50/AnythingXL_v50.safetensors"
    pipe = StableDiffusionPipeline.from_single_file(model_id)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("./model/lora/", weight_name="J_illustration.safetensors")
    pipe = pipe.to("cuda")

    image = pipe(
        prompt,
        negative_prompt="easy_negative, NSFW",
        guidance_scale=7,
        # negative_prompt="nsfw, (worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), (blush:1.2), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature",
        num_inference_steps=total_step_gen,
        callback=progress_with_socketid,
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
    # global progress_percentage_user
    # progress_percentage_user = 0.0

    socketid = request.values.get("socketid")
    # print(socketid)
    animal = request.values.get("animal")
    style_ = request.values.get("style_")
    action_ = request.values.get("action_")

    prompt = (
        "master piece, high quality"
        + ", a "
        + style_
        + " "
        + animal
        + ","
        + action_
        + "<lora:J_illustration:0.8> j_illustration"
    )
    # print("prompt:", prompt)
    image = generate_image(prompt,socketid)
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

    # global progress_percentage_user
    # progress_percentage_user = 0.0
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

# progress_dict = {}

def progress(step, timestep, latents,socketid):

    global matchsocketid
    matchsocketid = socketid
    print(socketid)
    
    print(float((step / total_step_gen)))
    progress_percentage_user[socketid] = float((step / total_step_gen))
    # global progress_percentage_user
    # progress_percentage_user = float((step / total_step_gen))


progress_dict = {}

@app.route("/progressInfo/<socketid>", methods=["POST"])
async def progressInfo(socketid):
    
    # Initialize progress for this socketid if it doesn't exist
    progress_dict[socketid] = 0
    progress_percentage_user[socketid] = 0 
    # if socketid not in progress_dict:
    #     progress_dict[socketid] = 0
    
    print("INIT",socketid,progress_dict[socketid])
    # if progress_dict[socketid] >= 0.9:
    #     progress_dict[socketid] = 0
    

    while progress_dict[socketid] < 0.9:
        # Update progress
        progress_dict[socketid] = progress_percentage_user.get(socketid, 0)

        # if(progress_dict[socketid]>0.9):
        #     progress_dict[socketid] = 1
        # progress_dict[socketid] = progress_percentage_user
        print(socketid,progress_dict[socketid])
        socketio.emit("update progress", progress_dict[socketid], to=socketid)
        await sleep(0.1)
    
    # When finished, ensure progress is set to 1
    print(socketid,progress_dict[socketid])
    progress_dict[socketid] = 1
    socketio.emit("update progress", progress_dict[socketid], to=socketid)

    return Response(status=204)



@app.route("/")
def index():
    return render_template("index.html")

def main():
    print("Starting server...")
    # app.run(host="localhost", port=5000)
    socketio.run(app, host="localhost", port=5000)


if __name__ == "__main__":
    main()


