Stable Diffusion Image Generation
======================================================

A simple API that generates images from text prompts using stable diffusion. The main goal of this project is to explore stable diffusion techniques for image generation and provide a user-friendly interface for generating images by providing a text prompt. This API is lightweight, easy to use, and can be a great starting point for learning about text-to-image generation 

Table of Contents
-----------------

*   [Requirements](#requirements)
*   [Examples](#examples)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Endpoints](#endpoints)

Requirements
------------

* Python 3.7 or higher
* Pipenv
* A stable internet connection for downloading the model (approx. 10GB)
* A GPU for running the model (tested with Nvidia 2060 Super)


Examples
--------

To generate an image using the API, send a GET request with the `prompt` query parameter:

```perl
http://localhost:5000/generate?prompt=Beautiful%20sunset%20over%20the%20ocean
```

This will return a generated image of a beautiful sunset over the ocean.

![Screenshot](https://raw.githubusercontent.com/prakashrx/stable-diffusion/main/images/screenshot.jpg)

Installation
------------

1.  Clone the repository:
    
    bash
    
    ```bash
    git clone https://github.com/prakashrx/stable-diffusion.git
    cd stable-diffusion
    ```
    
2.  Install the dependencies using Pipenv:
    
    ```bash
    pip install pipenv
    pipenv install
    ```

Usage
-----

1.  Activate the virtual environment:
    
    `pipenv shell`
    
2.  Run the Flask app:
    
    `python app.py`
    
3.  The API will be available at `http://localhost:5000/generate`.
    

Endpoints
---------

### `/generate`

*   Method: `GET`
*   Query Parameters:
    *   `prompt` (required): The text prompt to generate an image from.
*   Response: An image in JPEG format.
