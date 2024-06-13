Tugas pengolahan citra pertemuan ke-14

 Nama : Hansen Julio
 
 Kelas : TI.22.A5
 
NIM : 312210523

# import module
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# assign and open image
url = 'https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcS6HuPXKLP6UfXBrzMz42_2w-8nPjgCVZNmoA2AcNt_KXR8vcMdMra-IijGBznEsxwEFIwzRSRTlRE5kDM'
response = requests.get(url, stream=True)

with open('image.png', 'wb') as f:
    f.write(response.content)

img = cv2.imread('image.png')

# Converting the image into gray scale for faster
# computation.
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculating the SVD
u, s, v = np.linalg.svd(gray_image, full_matrices=False)

# inspect shapes of the matrices
print(f'u.shape:{u.shape}, s.shape:{s.shape}, v.shape:{v.shape}')

# Variance Explained
var_explained = np.round(s**2 / np.sum(s**2), decimals=6)

# Variance explained top Singular vectors
print(f'Variance Explained by Top 20 singular values:\n{var_explained[0:20]}')

sns.barplot(x=list(range(1, 21)), y=var_explained[0:20], color="dodgerblue")

plt.title('Variance Explained Graph')
plt.xlabel('Singular Vector', fontsize=16)
plt.ylabel('Variance Explained', fontsize=16)
plt.tight_layout()
plt.show()

# plot images with different number of components
comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    low_rank = u[:, :comps[i]] @ np.diag(s[:comps[i]]) @ v[:comps[i], :]

    if i == 0:
        plt.subplot(2, 3, i+1)
        plt.imshow(low_rank, cmap='gray')
        plt.title(f'Actual Image with n_components = {comps[i]}')

    else:
        plt.subplot(2, 3, i+1)
        plt.imshow(low_rank, cmap='gray')
        plt.title(f'n_components = {comps[i]}')

plt.show()
