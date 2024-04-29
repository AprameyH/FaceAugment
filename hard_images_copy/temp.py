import os

print(os.getcwd())
path = "C:\\Users\\tyler\\OneDrive\\Documents\\FaceAugment\\hard_images_copy\\Yui_Aragaki"
files = os.listdir(path)

for index, filename in enumerate(files):
    new_name = 'ya' + str(index) + '.jpg'
    os.rename(os.path.join(path, filename), os.path.join(path, new_name))