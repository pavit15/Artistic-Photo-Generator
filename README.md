# Images Generator

This project is an extension of my previous work on image generation using Stable Diffusion, available here:  
[Image-Generation-using-Stable-Diffusion](https://github.com/pavit15/Image-Generation-using-Stable-Diffusion)

In this extension, I focused on applying image generation techniques for specific artistic use cases such as Comic Book, Watercolor, Anime styles, and more, to create a reliable cartoon transformation pipeline with optimized preprocessing and a user friendly interface.

## Features

- Multiple artistic styles: Comic Book, Anime, Watercolor, 3D Cartoon, Sketch, Oil Painting
- Adjustable transformation strength for fine control over cartoonization
- Fast and optimized preprocessing with OpenCV-based edge preserving filters


## Architecture of the project
1. Upload a photo to the web interface.
<img src="https://github.com/pavit15/Artistic-Photo-Generator/blob/main/imgs/img1.jpg?raw=true" alt="Generated Artistic Image" width="600"/>

2. Preprocessing using edge preserving filters such as bilateral filter
This is done to smooth colors while keeping edges sharp which is necessary for the effects.
<img src="https://github.com/pavit15/Artistic-Photo-Generator/blob/main/imgs/img2.jpg?raw=true" alt="Generated Artistic Image" width="600"/>

4. Edge Detection & Combination
Edges are detected using adaptive thresholding and combined with the smoothed image to enhance outlines, mimicking the art.

5. Stable Diffusion Img2Img Pipeline
The preprocessed image is passed to the Stable Diffusion v1.5 model via an image-to-image pipeline, guided by the input art style chosen(comic book, anime, watercolor, etc.).
<img src="https://github.com/pavit15/Artistic-Photo-Generator/blob/main/imgs/img3.jpg?raw=true" alt="Generated Artistic Image" width="600"/>

7. Style Transformation & Output
The model generates a styled image. The result is resized back to the original aspect ratio and displayed in the interface.
<img src="https://github.com/pavit15/Artistic-Photo-Generator/blob/main/imgs/img4.jpg?raw=true" alt="Generated Artistic Image" width="600"/>

