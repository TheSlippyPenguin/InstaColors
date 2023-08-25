import cv2
import numpy as np
import instaloader
import os
import datetime

POST_SIZE = 100
USERNAME = "asphalte_homme"
DAYS = 90
COLOR_COUNT = 5

def concat_posts(posts):
    output = np.zeros((POST_SIZE, POST_SIZE * len(posts), 3), dtype=np.uint8)
    for i in range(len(posts)):
        output[:, i*POST_SIZE:i*POST_SIZE+POST_SIZE] = posts[i]
    return output   

def kmeans_colors(image, color_count=4):
    
    Z = image.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = color_count
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)

    colors = []
    for i in range(K):
        colors.append(  (center[i][0], center[i][1], center[i][2])  )
    
    return colors

# Load posts from username
loader = instaloader.Instaloader()
profile = instaloader.Profile.from_username(loader.context, USERNAME)
#filter by date
loader.download_profiles([profile], profile_pic=False, fast_update=True, post_filter=lambda post: post.date_utc > datetime.datetime.now() - datetime.timedelta(days=DAYS))

# open username directory and get all posts jpg images
folder_path = USERNAME
jpg_images = [file for file in os.listdir(folder_path) if file.lower().endswith(".jpg")]

# Make cv2 array of every post
post_images = []
for image in jpg_images:
    post_cv2 = cv2.imread(os.path.join(folder_path, image))
    post_cv2 = cv2.resize(post_cv2, (POST_SIZE, POST_SIZE) )
    post_images.append(post_cv2)

output_image = concat_posts(post_images)

colors = kmeans_colors(output_image, color_count=COLOR_COUNT)

# make image with 4 colors
color_image = np.zeros((100, 100 * len(colors), 3), dtype=np.uint8)
for i in range(len(colors)):
    color_image[:, i*100:i*100+100] = colors[i]
    
cv2.imwrite(f"{USERNAME}.jpg", color_image)

# delete username directory
for file in os.listdir(folder_path):
    os.remove(os.path.join(folder_path, file))
os.rmdir(folder_path)

print("Done")