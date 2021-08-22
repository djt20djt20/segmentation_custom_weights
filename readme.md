The purpose of this repo was to learn how to slice up image files to do segmentation (easy), and also to learn how to apply my own different weights to different pixels 
depending on what class they are (hard). Here's the order I do things:

1. Chop the images up smaller images
2. Define the unet model (making sure the last layer is one-hot encoded)
3. Train the model, using a custom loss function 

I didn't get around to stitichin
