""" This program stores and captures images/slices from CT scan videos provided by JIPMER"""
import cv2
import os

## User Defined Inputs
# Number of frames to be captured per second 
FPS = 3
# Parent directory of the directory containing all CT scan videos
SPATH = os.getcwd()
# Name of directory containing CT scan videos
DIR = 'Scan Videos'


# Create a directory to store the images
try:
    if not os.path.exists('all_images'):
        os.makedirs('all_images')
    else: quit()
except OSError:
    print ('Error: Creating directory of data')


# Find names of all video files with extension .avi
def video_crawler(SPATH):
    return [video_file for _, _, files in os.walk(SPATH) for video_file in files if '.avi' in video_file]


files = video_crawler(SPATH)

# Find all frames in video and write the frames to the newly created images directory
for video_file in files:

    location = SPATH + f'\\{DIR}\\{video_file[:-4]}\\Video\\AVI\\{video_file}'
    cap = cv2.VideoCapture(location)

    # Number of Images in a video_file
    count = 0

    fps = cap.get(cv2.CAP_PROP_FPS)     # video has 24 frames per second

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # Write FPS(here 3) images every second
            if ((count*FPS) % fps) == 0:
                name = f'./all_images/{video_file[:-4]}--' + str(int((count*FPS)/24)) + '.jpg'
                print ('Creating...' + name)
                cv2.imwrite(name, frame)
            count = count + 1
        else: break

cap.release()
cv2.destroyAllWindows()


