import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


app = FaceAnalysis(name='buffalo_l',  providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640,640))

swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=False, download_zip=False)
# main_img: File Path to AI generated background image (with one person)
# face_image: File path to original/celebrity face
# main_img: File Path to AI generated background image (with one person)
# face_image: File path to mugshot/celebrity face
def swapswap(main_img, face_image, show_steps=False, write=""):
    img2 = cv2.imread(main_img)
    
    if show_steps:
        plt.imshow(img2[:,:,::-1])
        plt.title('Background AI image')
        plt.show()

    img_face = cv2.imread(face_image)
    if show_steps:
        plt.imshow(img_face[:,:,::-1])
        plt.title('Original image')
        plt.show()

    face_out = app.get(img2)[0]

    face_in = app.get(img_face)[0]

    res2 = img2.copy()

    res2 = swapper.get(res2, face_out, face_in, paste_back=True)
    
    if write != "":
        plt.imsave(write, res2[:,:,::-1])

    return res2


res2 = swapswap(f'data/generated_images/haruna_1_generated/1024/HK0.png', '/Users/apramey/FaceAugment/data/final_images/all_haruna_source/haruna_1.jpg', show_steps=True)
plt.imshow(res2[:,:,::-1])
plt.show()