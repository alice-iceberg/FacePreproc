import os
import cv2
import dlib
import numpy as np
from skimage import draw
from scipy.spatial.qhull import ConvexHull
from PIL import Image


def load_images_from_folder(folder):
    print('Reading images...')
    original_images = []
    original_names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        basename = os.path.splitext(os.path.basename(filename))[0]
        if basename is not None:
            original_images.append(img)
            original_names.append(basename)
    return original_images, original_names


ready_images = []
# haarcascade_frontalface_alt2 classifier detected most faces
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')  # haarcascade_frontalface_default.xml lbpcascade_profileface.xml
count = 0

# #################################PART1#####################################
# #################################Face crop#################################
# input_names = load_images_from_folder("mouth_closed")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# for i in range(len(input_names)):
#     img = dlib.load_rgb_image("mouth_closed/" + input_names[i] + ".jpg")
#     rect = detector(img)[0]
#     sp = predictor(img, rect)
#     landmarks = np.array([[p.x, p.y] for p in sp.parts()])
#
#     # Select the landmarks that represents the shape of the face
#     outline = landmarks[[*range(17), *range(26, 16, -1)]]
#
#     # Draw a polygon using these landmarks using scikit-image
#     Y, X = draw.polygon(outline[:, 1], outline[:, 0])
#
#     # Create a canvas with zeros and use the polygon as mask to original image
#     cropped_img = np.zeros(img.shape, dtype=np.uint8)
#     cropped_img[Y, X] = img[Y, X]
#
#     vertices = ConvexHull(landmarks).vertices
#     Y, X = draw.polygon(landmarks[vertices, 1], landmarks[vertices, 0])
#     cropped_img = np.zeros(img.shape, dtype=np.uint8)
#     cropped_img[Y, X] = img[Y, X]
#     im = Image.fromarray(cropped_img)
#     im.save("face_oval_closed/" + input_names[i] + ".jpg")
#     print("Image cropped: ", input_names[i])

################################PART2################################
# Recolor and reshape
input_images, input_names = load_images_from_folder("turabek")

for i in range(len(input_images)):
    gray_image = input_images[i]
    #blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    if len(faces) == 0:
        print(i, ' no face')
    elif len(faces) > 1:
        print(i, ' more than 1 face')
        continue

    else:
        print('Image is being processed: ', i, " ",  input_names[i])
        face = faces[0]
        x, y, w, h = [v for v in face]
        face_crop = gray_image[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (224, 224), cv2.INTER_LANCZOS4)
        normalizedImg = np.zeros((800, 800))
        normalizedImg = cv2.normalize(face_crop, normalizedImg, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('all_224/{}.jpg'.format(input_names[i]), normalizedImg)
        lin_img = normalizedImg.flatten()
        pixel_list = lin_img.tolist()
        pixel_str_list = map(str, pixel_list)
        img_str = ' '.join(pixel_str_list)
        ready_images.append(img_str)
        count += 1

print("Found {0} Faces!".format(count))
print("Gray images: {0}".format(len(ready_images)))
