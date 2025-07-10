from PIL import Image

img = Image.open("cat.jpg")

print(img.mode, img.size, img.format)
img.save("cat.png")

r, g, b = img.split()
# r.show()

img_rotated = img.rotate(90)
# img_rotated.show()

img_resize = img.resize((800, 400))
img_resize.show()
