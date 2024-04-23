from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler
)

from flask import Flask, Response, request, jsonify, render_template, send_file
from flask_cors import CORS
import io
import psutil
import pynvml
from flask_socketio import SocketIO
from asyncio import sleep
import os

total_step_gen = 40
finished = False
app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
progress_percentage_user = {}


def generate_image(prompt, socketid):
    def progress_with_socketid(step, timestep, latents):
        progress(step, timestep, latents, socketid)

    # model_id = "./AnythingXL_v50/AnythingXL_v50.safetensors"
    # pipe = StableDiffusionPipeline.from_single_file(model_id)
    repo_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("./model/lora/", weight_name="J_illustration.safetensors")

    pipe = pipe.to("cuda")
    image = pipe(
        width=512,
        height=768,
        prompt=prompt,
        negative_prompt="easy_negative, NSFW",
        guidance_scale=7,
        num_inference_steps=total_step_gen,
        callback=progress_with_socketid,
        callback_steps=1,
    ).images[0]
    return image


@app.route("/generate", methods=["POST"])
def generate():
    socketid = request.values.get("socketid")
    animal = request.values.get("animal")
    style_ = request.values.get("style_")
    action_ = request.values.get("action_")
    weather = request.values.get("weather")
    place = request.values.get("place")
    season_ = request.values.get("season_")
    time_of_day = request.values.get("time_of_day")
    perspective = request.values.get("perspective")

    prompt = (
        "master piece, high quality"
        + ","
        + style_
        + ","
        + animal
        + ","
        + weather
        + ","
        + place
        + ","
        + action_
        + ","
        + season_
        + ","
        + time_of_day
        + ","
        + perspective
        + ","
        + "<lora:J_illustration:0.8> j_illustration"
    )
    image = generate_image(prompt, socketid)
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
    
def progress(step, timestep, latents, socketid):

    global matchsocketid
    matchsocketid = socketid
    print(socketid)

    print(float((step / total_step_gen)))
    progress_percentage_user[socketid] = float((step / total_step_gen))


progress_dict = {}


@app.route("/progressInfo/<socketid>", methods=["POST"])
async def progressInfo(socketid):

    progress_dict[socketid] = 0
    progress_percentage_user[socketid] = 0

    print("INIT", socketid, progress_dict[socketid])

    while progress_dict[socketid] < 0.9:
        progress_dict[socketid] = progress_percentage_user.get(socketid, 0)
        print(socketid, progress_dict[socketid])
        socketio.emit("update progress", progress_dict[socketid], to=socketid)
        await sleep(0.1)

    print(socketid, progress_dict[socketid])
    progress_dict[socketid] = 1
    socketio.emit("update progress", progress_dict[socketid], to=socketid)

    return Response(status=204)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/monitor_data")
def monitor_data():

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

    info = {
        "cpu_usage": psutil.cpu_percent(interval=1),
        "ram_usage": psutil.virtual_memory().percent,
        "gpu_usage": gpu_util,
    }
    return jsonify(info)


@app.route("/restart")
def restart():
    os.system("killall -9 python; python app.py &")
    return "Server is restarting..."


def main():
    print("Starting server...")
    socketio.run(app, host="0.0.0.0", port=10232)


if __name__ == "__main__":
    main()
