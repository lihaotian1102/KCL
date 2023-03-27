import os
from PIL import Image

# Create the frames
frames = []

path = "../path"
for i in range(1, 100):
    new_frame = Image.open(path + "/cpath" + str(i) + "_pic.png")
    frames.append(new_frame)

# Save into a GIF file
frames[0].save(path + "/gif/rrtc_t10"
                      "_animation.gif", format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1, transparency=10)