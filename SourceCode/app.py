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
import psutil
import pynvml
from flask_socketio import SocketIO
from werkzeug.exceptions import BadRequest
from asyncio import sleep
from user import User
import os
import tempfile

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
        num_inference_steps=total_step_gen,
        callback=progress_with_socketid,
        callback_steps=1,
    ).images[0]

    # print(image.hidden_states)
    return image


@app.route("/generate", methods=["POST"])
def generate():
    socketid = request.values.get("socketid")
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


@app.route("/iti", methods=["POST"])
def imgtoimg():
    try:
        socketid = request.values.get("socketid")
        image_byte = request.form['img_byte']
        # test = request.files['img_file']
        
        # print(test==None)
        # print(type(test.stream))
        # test.seek(0)
        # image = Image.open(test)
        # temp_file.file.seek(0)  # change the position to the beginning of the file
        # f = {'file': (temp_file.name, temp_file, 'png')}
        return jsonify(socket=socketid,testing=image_byte)

    except BadRequest:
        return jsonify(error='The file is too large'), 400
    except Exception as e:
        return jsonify(error=str(e)), 400


def progress(step, timestep, latents, socketid):

    global matchsocketid
    matchsocketid = socketid
    print(socketid)

    print(float((step / total_step_gen)))
    progress_percentage_user[socketid] = float((step / total_step_gen))


progress_dict = {}


@app.route("/progressInfo/<socketid>", methods=["POST"])
async def progressInfo(socketid):

    # Initialize progress for this socketid if it doesn't exist
    progress_dict[socketid] = 0
    progress_percentage_user[socketid] = 0
    # if socketid not in progress_dict:
    #     progress_dict[socketid] = 0

    print("INIT", socketid, progress_dict[socketid])
    # if progress_dict[socketid] >= 0.9:
    #     progress_dict[socketid] = 0

    while progress_dict[socketid] < 0.9:
        # Update progress
        progress_dict[socketid] = progress_percentage_user.get(socketid, 0)

        # if(progress_dict[socketid]>0.9):
        #     progress_dict[socketid] = 1
        # progress_dict[socketid] = progress_percentage_user
        print(socketid, progress_dict[socketid])
        socketio.emit("update progress", progress_dict[socketid], to=socketid)
        await sleep(0.1)

    # When finished, ensure progress is set to 1
    print(socketid, progress_dict[socketid])
    progress_dict[socketid] = 1
    socketio.emit("update progress", progress_dict[socketid], to=socketid)

    return Response(status=204)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/monitor")
def monitor():

    return render_template("monitor.html")


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
    # app.run(host="localhost", port=5000)
    socketio.run(app, host="localhost", port=5000)


if __name__ == "__main__":
    main()
