from pathlib import Path
import numpy as np
import cv2

path = r"D:\Datasets\flood_area_segmentation\Image"  # for windows

imageFilesDir = Path(path)
files = list(imageFilesDir.rglob('*.jpg'))

# Since the std can't be calculated by simply finding it for each image and averaging like  
# the mean can be, to get the std we first calculate the overall mean in a first run then  
# run it again to get the std.

mean = np.array([0.,0.,0.])
stdTemp = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])

numSamples = len(files)
skippedSamples = 0

print(cv2.imread(str(files[0])))

for i in range(numSamples):
    im = cv2.imread(str(files[i]))

    if im is None:
        skippedSamples += 1
        continue

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    
    for j in range(3):
        mean[j] += np.mean(im[:,:,j])

mean = (mean/(numSamples - skippedSamples))

for i in range(numSamples):
    im = cv2.imread(str(files[i]))

    if im is None:
        skippedSamples += 1
        continue

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255.
    for j in range(3):
        stdTemp[j] += ((im[:,:,j] - mean[j])**2).sum()/(im.shape[0]*im.shape[1])

std = np.sqrt(stdTemp/(numSamples - skippedSamples))

print(f"mean = {mean}")
print(f"std = {std}")


