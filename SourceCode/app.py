# Path: app.py
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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
finished = False
app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)

# model_id = "stabilityai/stable-diffusion-2-1"
model_id = "WarriorMama777/AbyssOrangeMix2"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.safety_checker = lambda images, clip_input: (images, False)
pipe = pipe.to("cuda")


# pipe.enable_attention_slicing()


def generate_image(prompt):
    image = pipe(
        prompt,
        negative_prompt="nsfw, (worst quality, low quality:1.4), (lip, nose, tooth, rouge, lipstick, eyeshadow:1.4), (blush:1.2), (jpeg artifacts:1.4), (depth of field, bokeh, blurry, film grain, chromatic aberration, lens flare:1.0), (1boy, abs, muscular, rib:1.0), greyscale, monochrome, dusty sunbeams, trembling, motion lines, motion blur, emphasis lines, text, title, logo, signature",
        num_inference_steps=30,
        callback=progress,
        callback_steps=1,
    ).images[0]
    # print(image.hidden_states)
    return image


@app.route("/generate", methods=["POST"])
def generate():
    global progress_percentage_user
    progress_percentage_user = 0.0
    global finished
    finished = False
    socketid = request.args.get("socketid")
    prompt = request.values.get("prompt")

    # neagative_prompt = request.args.get("neagative_prompt")
    print("prompt:", prompt)
    image = generate_image(prompt)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    finished = True
    # print(f"Generated image of size {len(image_bytes)} bytes.")

    # encoded_img_data = base64.b64encode(image.getvalue())

    # return the response as an image
    # return Response(image_bytes, mimetype="image/jpeg")
    # return render_template("home.html", user_image = image_bytes)
    return send_file(
        io.BytesIO(image_bytes),
        mimetype="image/jpeg",
        download_name="img.jpg",
        as_attachment=True,
    )
    # return send_file(BytesIO(resp.content), mimetype="image/jpeg", attachment_filename="img2.jpg", as_attachment=True)


def progress(step, timestep, latents):

    global progress_percentage_user
    progress_percentage_user = float((step / timestep) * 100)
    # progress_percentage1 = progress_percentage
    print(f"Progress: {progress_percentage_user}%")


@app.route("/progressInfo/<socketid>", methods=["POST"])
async def progressInfo(socketid):

    while not finished:

        print(progress_percentage_user)
        socketio.emit("update progress", progress_percentage_user, to=socketid)
        await sleep(0.3)

    socketio.emit("update progress", 100, to=socketid)
    # finished = False;
    return Response(status=204)


@app.route("/")
def index():
    return render_template("home.html")

@app.route("/on9")
def on9():
    return render_template("x.html")


# Start an http server and expose an api endpoint that takes in a prompt and returns an image.
def main():
    print("Starting server...")
    # app.run(host="localhost", port=5000)
    socketio.run(app, host="localhost", port=5000)


if __name__ == "__main__":
    main()
