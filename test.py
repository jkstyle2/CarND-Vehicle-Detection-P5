from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time

# %matplotlib inline

print('...')


car_images = glob.glob('dataset/vehicles/**/*.png')
noncar_images = glob.glob('dataset/non-vehicles/**/*.png')

print("The number of vehicle dataset: ", len(car_images))
print("The number of non-vehicle dataset: ", len(noncar_images))

figure, axs = plt.subplots(6, 6, figsize=(16, 16))
figure.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()  # change to a 1d array to handle simply

# visualize car images
for i in np.arange(18):
    random_index = np.random.randint(0, len(car_images))
    image = cv2.imread(car_images[random_index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axs[i].imshow(image)
    axs[i].axis('off')
    axs[i].set_title('car', fontsize=11)

# visualize noncar images
for i in np.arange(18, 36):
    random_index = np.random.randint(0, len(noncar_images))
    image = cv2.imread(noncar_images[random_index])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axs[i].imshow(image)
    axs[i].axis('off')
    axs[i].set_title('noncar', fontsize=11)


def get_hog_features(image, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(image, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False, visualize=vis,
                                  feature_vector=feature_vec)
        return features, hog_image

    # Otherwise, call with one output
    else:
        features = hog(image, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False, visualize=vis,
                       feature_vector=feature_vec)
        return features


print('...')

car_image = mpimg.imread(car_images[5])
_, car_hog = get_hog_features(car_image[:, :, 2], orient=9, pix_per_cell=8, cell_per_block=8, vis=True, feature_vec=True)

noncar_image = mpimg.imread(noncar_images[5])
_, noncar_hog = get_hog_features(noncar_image[:, :, 2], orient=9, pix_per_cell=8, cell_per_block=8, vis=True, feature_vec=True)

# Visualize HOG features
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(7,7))
figure.subplots_adjust(hspace=.4, wspace=.2)

ax1.imshow(car_image)
ax1.axis('off')
ax1.set_title('Car', fontsize=15)
ax2.imshow(car_hog, cmap='gray')
ax2.axis('off')
ax2.set_title('Car HOG', fontsize=15)
ax3.imshow(noncar_image)
ax3.axis('off')
ax3.set_title('Non-Car', fontsize=15)
ax4.imshow(noncar_hog, cmap='gray')
ax4.axis('off')
ax4.set_title('Non-Car HOG', fontsize=15)


# Define a function to extract features from a list of image locations
# This function could also be used to call bin_spatial() and color_hist() (as in the lessons)
# to extract flattened spatial color features and color histogram features and combine them all (making use of StandardScaler)
# to be used together for classification
def extract_features(images, color_space='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in images:
        # Read in each one by one
        image = mpimg.imread(file)
        # Apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_feature = get_hog_features(feature_image[:, :, channel], orient, pix_per_cell, cell_per_block,
                                               vis=False, feature_vec=True)  # type: list
                hog_features.append(hog_feature)  # type: list
            hog_features = np.ravel(hog_features)  # type: ndarray (change a list to 1d ndarray)

        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell, cell_per_block,
                                            vis=False, feature_vec=True)

        # Append the new feature vector to the features list
        features.append(hog_features)
    # Return list of feature vectors
    return features


print("...")




# little confused about data type. check this out

a = np.array([1,2,3])
print("a: ", a)
print("a.type: ",type(a))
a2 = []
a2.append(a)
print("a2: ", a2)
print("a2.type: ", type(a2))
a3 = np.ravel(a2)
print("a3: ", a3)
print("type(a3): ", type(a3))




# Feature extraction parameters
colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or 'ALL'

t = time.time()
car_features = extract_features(car_images, color_space=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
notcar_features = extract_features(noncar_images, color_space=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
t2 = time.time()
print(round(t2 - t, 2), "seconds to extract HOG features...")

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per-column scaler - this will be necessary if combining different types of features (HOG + color_hist + bin_spatial)
# X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
# scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler - this will be necessary if combining different types(scale) of features (HOG + color_hist + bin_spatial)
# X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X
# X_train = X_scaler.transform(X_train)
# X_test = X_scaler.transform(X_test)

print("Using:", colorspace,"Colorspace,", orient, 'orientations,', pix_per_cell, 'pixels/cell, ', cell_per_block, 'cells/block')
print("Feature vector length: ", len(X_train[0]))




# Use a linear SVC
svc = LinearSVC()

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'seconds to train SVM Classifier.')
print("...")

# Check the score of the SVC
svc_score = round(svc.score(X_test, y_test), 4)
print("Test Accuracy of SVM =", svc_score)
print("...")

# Check the prediction time for a single sample
n_predict = 10
t = time.time()
svc_predict = svc.predict(X_test[0:n_predict])
print("For these TRUE Labels: ", y_test[0:n_predict])
print("The SVM Classifier prediction result:  ", svc_predict)
t2 = time.time()
print(round(t2 - t, 6), "seconds to predict", n_predict, "labels with the SVM Classifier.")


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(image, ystart, ystop, scale, color_space, hog_channel, svc, X_scaler, orient,
              pix_per_cell, cell_per_block, spatial_size, hist_bins, show_all_rectangles=False):
    # arrays of rectangles where cars were detected
    rectangles = []
    image = image.astype(np.float32) / 255
    image_tosearch = image[ystart:ystop, :, :]

    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            color_transform_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            color_transform_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            color_transform_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            color_transform_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            color_transform_tosearch = cv2.cvtColor(image_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        color_transform_tosearch = np.copy(image)

    # Rescale image if other than 1.0 scale
    if scale != 1:
        image_shape = color_transform_tosearch.shape
        color_transform_tosearch = cv2.resize(color_transform_tosearch, (np.int(image_shape[1] / scale), np.int(image_shape[0] / scale)))

    # Select color space channel for HOG
    if hog_channel == 'ALL':
        ch1 = color_transform_tosearch[:, :, 0]
        ch2 = color_transform_tosearch[:, :, 1]
        ch3 = color_transform_tosearch[:, :, 2]
    else:
        ch1 = color_transform_tosearch[:, :, hog_channel]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    if hog_channel == 'ALL':
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # Extract HOG fot this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = hog_feat1

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            ############# IN CASE OF USING OTHER FEATURE EXTRACTION ###########

            #             # Extract the image patch
            #             subimg = cv2.resize(color_transform_tosearch[ytop:ytop+window , xleft:xleft+window], (64,64))

            #             # Get color features
            #             spatial_features = bin_spatial(subimg, size=spatial_size)
            #             hist_features = color_hist(subimg, nbins=hist_bins)

            #             # Scale features and make a prediction
            #             test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))
            #             test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1,-1))
            #             test_prediction = svc.predict(test_features)

            ##############################################################

            test_prediction = svc.predict(hog_features.reshape(1, -1))  # only use hog feature

            if test_prediction == 1 or show_all_rectangles:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(
                    ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return rectangles


print("...")




test_image = mpimg.imread('./test_images/test1.jpg')

ystart = 400
ystop = 656
scale = 2
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or 'ALL'

rectangles = find_cars(test_image, ystart, ystop, scale, color_space, hog_channel, svc, None, orient, pix_per_cell, cell_per_block, None, None)

print(len(rectangles), 'rectangles found in image')


# Here is your draw_boxes function from the previous exercise
def draw_boxes(image, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(image)
    random_color = False

    # Iterate through the bounding boxes
    for bbox in bboxes:
        # for random color bboxes, select the color
        if color == 'random' or random_color:
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            random_color = True

        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)

    # Return the image copy with boxes drawn
    return imcopy


print("...")



test_image_rects = draw_boxes(test_image, rectangles)
plt.figure(figsize=(10,10))
plt.imshow(test_image_rects)
print("...")