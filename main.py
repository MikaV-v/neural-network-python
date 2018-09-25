import math
import image
import random
import PIL
import os

import Image, ImageDraw
import os
image = Image.open("red_410.png")
width = image.size[0]
height = image.size[1]
pix = image.load()
a = pix[4,4][0]
b = pix[4,4][1]
c = pix[4,4][2]
if (a+b+c<400):
    print('R=',a,'G=',b,'B=',c)
    print (os.path.realpath("red_410.png") )
else:
    print("Noooo")