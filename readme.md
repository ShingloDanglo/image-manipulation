# Usage
### Note
This tool is far from ready and should not be used in its current state.

### Image Example
The following example shows the workflow one might use for the creation of a seamless texture for use in 3d. The command below will resize an image to 1024x1024 and remove the seams that would appear when tiling. Image processes are run from left to right, so resizing the image should come first to reduce the time it takes to perform subsequent processes.

`
py main.py image path/to/input.png path/to/output.png resize make-seamless
`

Before:

<img width="25%" height="25%" alt="before-example" src="https://github.com/user-attachments/assets/8e38f2a7-ed11-43f9-8c1f-19bc74c49d97" />

After:

<img width="25%" height="25%" alt="after-example" src="https://github.com/user-attachments/assets/1136b1bc-a6cf-4031-ae0c-e62b9e153f02" />

### Video Example
Currently, videos can only have one process applied to them. All image processes are available.

`
py main.py video path/to/input.mp4 path/to/output.mp4 <process>
`

### GIF Example
MP4s can be converted to GIFs, with one image process applied. All image processes are available.
`
py main.py gif path/to/input.mp4 path/to/output.mp4 <process>
`

Here is a list of currently supported image processes and their associated command-line arguments:
- Ordered Dithering: "dither"
- Resizing: "resize"
- Gaussian Blur: "gaussian-blur"
- Box Blur: "box-blur"
- Posterization/quantization: "posterize"
- Sobel Edge Detection: "sobel-edge-detect"
- Make seamless: "make-seamless"

### Planned features
- GUI interface for ease of use
- More image processes
- Options to specify colour depth and colour palette
- Allow multiple image processes to be applied to videos and GIFs
