import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as pimg
from skimage.transform import pyramid_gaussian
from skimage.color.colorconv import rgb2gray
from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Parameters Setups
NUM_EIGEN_FACES = 100
K = 10
image_size = None
middle_point = 0
RGB = False # Set True for Colorful Eigenfaces 
test_names = []

# Folder Paths
TRAIN_DIR_T = "att_faces"
TRAIN_DIR_F = "256_ObjectCategories"
TEST_DIR = "tests_T"
AGE_TRAIN_DIR = "age_train"
AGE_TEST_DIR = "AgeTest"
AGE_RESULT = "age_result.txt"
MULTI_FACE = "test_multi/multi12.jpg"

# Convert RGB to GrayScale 
def img2gray(img):
    if len(img.shape) > 2 :
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        return img

# Convert CV2 BGR to skimage RGB
def bgr2rgb(img):
    b,g,r = cv2.split(img)
    return cv2.merge([r,g,b])

# read images from the path
def readImages(dir_path):
    global image_size, test_names
    
    print("Reading images from " + dir_path, end = " --- ")
    images = []
    for path in sorted(os.listdir(dir_path)):
        if path == ".DS_Store": continue
        
        if dir_path == TEST_DIR: test_names.append(path)
        img_path = os.path.join(dir_path, path)
        raw_image = pimg.imread(img_path) if RGB else img2gray(pimg.imread(img_path))
        if image_size == None: image_size = raw_image.shape
        # keep the same size
        image = cv2.resize(raw_image, (image_size[1], image_size[0]))
        
        images.append(image)
   
    image_num = len(images)
    if image_num == 0:
        print("No images found")
        sys.exit(0)
    print(str(image_num) + " images read.")
    
    color = 3 if RGB else 1
    
    flatten_images = np.zeros((image_num, image_size[0] * image_size[1] * color), dtype=np.float64)
    for i in range(image_num):
        flatten_images[i,:] = (images[i] / 255.0).flatten()
    return flatten_images, image_num
    
# Read the age from the file name
def readAge(dir_path):
    ages = []
    for path in sorted(os.listdir(dir_path)):
        age = int(path.split("_")[0])
        group = None
        if age < 20:
            group = '0-20 years old' 
        elif age < 40:
            group = '20-40 years old'
        elif age < 60:
            group = '40-60 years old'
        elif age < 80:
            group = '60-80 years old'
        else:
            group = '> 80 years old'
        ages.append(group)
    return ages


# Calculation for PCA
def PCA(images):
    mean = np.mean(images,axis=0)
    mean_sub = images - mean
    _, S, V = np.linalg.svd(mean_sub, full_matrices=False)
    eigen_vectors = V[:NUM_EIGEN_FACES]
    variance = ((S ** 2) / (len(images) - 1))
    print("Using {0:.2f}% of Variance".format(variance[:NUM_EIGEN_FACES].sum() * 100.0 / variance.sum()))
    return mean, variance, eigen_vectors

# Convert the flatten np.array to image shape
def convertFace(image):
    return image.reshape(image_size)

# Get the eigenfaces weights for input images
def getWeights(images, mean, eigen_vectors):
    mean_sub = images - mean
    weights = np.dot( mean_sub, eigen_vectors.T)
    return weights 

# Get the classification results
def getResults(train, test, k = K, covariance = None):

    if covariance is not None: inv_cov = np.linalg.inv(covariance)
    
    # store the sum distance of between each test and KNN in training set
    results = []
    similarity = []
    for test_img in test:
        # store distance between test_img and each image in training set
        _distance = []
        for train_img in train:
            
            # get mahalanobis distance
            if covariance is None:
                _distance.append(euclidean(test_img, train_img))  
            else:
                _distance.append(mahalanobis(test_img, train_img, inv_cov))
#             _distance[j] = np.sqrt((( train_img - test_img ) ** 2).sum())
        _distance = np.array(_distance)
        knn_index = np.argpartition(_distance, range(k))[:k]

        k_result = knn_index < middle_point
        similarity.append("{0:.0f} %".format(k_result.sum()*100.0 / k))
        
        base = np.zeros(k)
        for i in range(k):
            base[i] = -1 if k_result[i] == False else 1
        weights = _distance[knn_index] ** -1
        w_result = base * weights
        
        results.append(bool(w_result.sum() > 0))
    return results, similarity

# plot the image in rgb or grayscale
def pltImg(image):
    if RGB:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap = "gray")

# show the average face and top 20 eigenfaces 
def show(mean, eigen_vectors, variance):
    
    # show mean face
    mean_face = convertFace(mean)
    pltImg(mean_face)
    plt.axis("off")
    plt.show()
    
    # show top 20 eigenfaces
    var_sum = variance.sum()
    plt.figure(figsize=(7, 6))
    for i in range(20):
        eigenFace = convertFace(eigen_vectors[i])
        var_ratio = (variance[i] / var_sum) *100
        plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
        plt.subplot(4,5,i+1)
        pltImg(eigenFace*255) 
        plt.title("EigenFace" + str(i+1) + "\n Ratio: {0:.2f}%".format(var_ratio))
        plt.axis("off")
    plt.show()
    
# show the test results and reconstructed faces
def test(mean, eigen_vectors, train_weights, covariance=None):
    print("Loading Testing Images...")
    test_images,_ = readImages(TEST_DIR)
    print("")
    
    test_weights = getWeights(test_images, mean, eigen_vectors)
    results,_ = getResults(train_weights, test_weights) if covariance is None else getResults(train_weights, test_weights, covariance=covariance)
    
    print(test_names)
    np.set_printoptions(suppress=True)
    print(results)
    
    for i in range(len(test_weights)):
        fig = plt.figure(0)
        fig.canvas.set_window_title('Result: {}'.format(results[i]))
        weights = test_weights[i]
        reconstruct = np.dot(weights, eigen_vectors)
        plt.subplot(1,2,1)
        pltImg(convertFace(test_images[i]))
        plt.title(test_names[i] + ": Original")
        plt.subplot(1,2,2)
        pltImg(convertFace(reconstruct + mean))
        plt.title(test_names[i] + ": Reconstruct")
        plt.axis("off")
        plt.show()
        
        
# take fitted svm to predict the age of input image
def agePrediction(clf, mean, eigen_vectors):

    print("Loading Age Testing Images...")
    age_img_test,_ = readImages(AGE_TEST_DIR)
    age_test_weights = getWeights(age_img_test, mean, eigen_vectors)
    names = sorted(os.listdir(AGE_TEST_DIR))
    if ".DS_Store" in names: names.remove(".DS_Store")
    names = [i + " -> " for i in names]
    print("")   
    
    result = clf.predict(age_test_weights)
    result = [ names[i] + result[i] for i in range(len(names))]
    
    f = open(AGE_RESULT, "w")
    for res in result:
        print(res) 
        f.write(res + "\n")
    f.close()

# Sliding window
def slide(image, step, window_size):
    
    # sliding window across the image
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
            
# Ref: Malisiewicz et al. Reduce face overlap
def non_max_suppression_fast(coords, overlapThresh=0.2):
    # if there are no boxes, return an empty list
    if len(coords) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    sim = coords[:, 4]
    boxes = coords[:, :4].astype("float")
 
    # initialize the list of picked indexes    
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    picked_box = boxes[pick].astype("int")
    picked_sim = sim[pick]
    result = []
    for i in range(len(pick)):
        result.append((*picked_box[i], picked_sim[i]))
    return result
   
# detect faces in mutiple scales
def multiTest(mean, eigen_vectors, train_weights, covariance=None):

    multi_image  = cv2.imread(MULTI_FACE)
    
    window_size = (int(image_size[1] / 2), int(image_size[0] / 2))
    down_scale = 1.2
    skip_pixels = 10
    
    coords = []
    for (i, resized) in enumerate(pyramid_gaussian(multi_image, downscale=down_scale, multichannel=True)):
        if (resized.shape[1] <= window_size[0] or resized.shape[0] <= window_size[1]
            or resized.shape[1] <= 200 or resized.shape[0] <= 200):
                break
        _coords = []
        for (x, y, window) in slide(resized, skip_pixels, window_size):
            if window.shape[::-1][1:] != window_size: continue
                
            mul_image = bgr2rgb(window) if RGB else rgb2gray(bgr2rgb(window))
            mul_image = cv2.resize(mul_image, (image_size[1], image_size[0])).flatten()

            test_img = []
            test_img.append(mul_image)
            
            # Classifier
            mul_weights = getWeights(test_img, mean, eigen_vectors)
            result,similarity = getResults(train_weights, mul_weights, covariance=covariance)
            
            
            if result[0]:
                _coords.append((x, y, similarity[0]))
                
            # Show detected face in current scale
            cv_copy = resized.copy()
            cv2.rectangle(cv_copy, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 1)
            for x, y, sim in _coords:
                cv2.rectangle(cv_copy, (x, y), (x + window_size[0], y + window_size[1]), (0, 0, 255), 1) 
                cv2.putText(cv_copy, sim, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Sliding Window", cv_copy)
            cv2.waitKey(1) 
          
        for x, y, sim in _coords:
            down_size = down_scale**i
            w = int(window_size[0] * down_size)
            h = int(window_size[1] * down_size)
            x, y = int(x * down_size), int(y * down_size)
            coords.append((x, y, x + w, y + h, sim))
    
    coords = non_max_suppression_fast(np.array(coords))
    # show all detected faces
    for [x1, y1, x2, y2, sim] in coords:
        
        cv2.rectangle(multi_image, (x1, y1), (x2, y2), (0, 255, 0), 1) 
        cv2.putText(multi_image, sim, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    cv2.imshow("Detected Faces", multi_image)
    cv2.waitKey(0)
    
# Collect detected faces from harr classifier
def webcam_train(name):

    haar = 'haarcascade_frontalface_default.xml'
    haar_cascade = cv2.CascadeClassifier(haar)
    
    if not os.path.isdir("data"):
        os.mkdir("data")
        
    data_dir = os.path.join("data", name) 
    num = 0
    
    # Check if the person is stored before
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    else:
        num = len(os.listdir(data_dir))
        
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
    counter = 0
    freq = 2
    train_num = 100
    while rval:
        key = cv2.waitKey(10)
        if key == 27 or num == train_num: # exit on ESC
            break
        
        rval, frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255 ,0),2)
            cv2.putText(frame, name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 3,(0, 0, 255), 3)
            face = cv2.resize(frame[y: y+h, x:x+w], (image_size[1], image_size[0]))
            
            if counter % freq == 0:
                cv2.imwrite('{}/{}.png'.format(data_dir, num), face)
                num += 1
                print('{}/{}.png Created'.format(data_dir, num))
                
        cv2.imshow("Training...", frame)
        

    vc.release()
    cv2.destroyWindow("preview")
    
    imgs = []
    indices = []
    i = 0
    for people in sorted(os.listdir("data")):
        if people ==".DS_Store":continue
        people_path = os.path.join("data", people)
        for img in sorted(os.listdir(people_path)):
            if img ==".DS_Store":continue
            img_path = os.path.join(people_path, img)
            imgs.append(cv2.imread(img_path, 0))
            indices.append(i)
        i += 1
    
    #Generating the data 
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(imgs, np.array(indices))
    model.write("saved_faces.xml")
    print("Training Complete!")
    
    
def webcam():
    haar = 'haarcascade_frontalface_default.xml'
    haar_cascade = cv2.CascadeClassifier(haar)
    
    model = cv2.face.EigenFaceRecognizer_create()
    model.read("saved_faces.xml")
    
    names = {}
    i = 0
    for people in sorted(os.listdir("data")):
        if people == ".DS_Store":continue
        names[i] = people
        i += 1
    
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        
    while rval:
        rval, frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (image_size[1], image_size[0]))
            result = model.predict(face) # predict face name
        
            name = "Unkown"
            if result[1]<3500:
                name = names[result[0]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255 ,0),2)
            cv2.putText(frame, name, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 255), 3)
        
        key = cv2.waitKey(10)
        if key == 27: # exit on ESC
            break
        cv2.imshow("Working...", frame)
        
    vc.release()
    cv2.destroyWindow("preview")
    
# show the performance of different number of eigenfaces M
def mStats(train_T, train_images):
    global NUM_EIGEN_FACES
    old = NUM_EIGEN_FACES
    M = [2, 5, 10, 20, 50, 100, 150, 200, 300]
    
    test_images_T,num_T = readImages("tests_T")
    test_images_F,num_F = readImages("tests_F")
    for i in range(2):
        x = []
        y = []
        for m in M:
            NUM_EIGEN_FACES = m
            mean, variance, eigen_vectors = PCA(train_T)
    
            train_weights_T = getWeights(train_T, mean, eigen_vectors)
            covariance = np.cov(train_weights_T.T) if i == 1 else None
    
            train_weights = getWeights(train_images, mean, eigen_vectors)
            
            test_weights_T = getWeights(test_images_T, mean, eigen_vectors)
            results_T,_ = getResults(train_weights, test_weights_T, covariance=covariance)
           
            test_weights_F = getWeights(test_images_F, mean, eigen_vectors)
            results_F,_ = getResults(train_weights, test_weights_F, covariance=covariance)
            
            accuracy = (np.array(results_T).sum() +  num_F - np.array(results_F).sum()) / (num_T + num_F)
            x.append(m)
            y.append(accuracy)
        if i == 0:
            label = "euclidean"
            marker = '.'
            color = 'red'
        else:
            label = "mahalanobis"
            marker = '*'
            color = 'blue'
        plt.plot(x, y, marker=marker, color=color, label=label, linewidth=2)
    
    plt.xlabel('Number of EigenFaces M')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    NUM_EIGEN_FACES = old
    
    
# show the performance of different number of neighbors k 
def kStats(mean, eigen_vectors, train_weights, covariance):
    ks = [1, 5, 10, 15, 20, 50]
    test_images_T,num_T = readImages("tests_T")
    test_images_F,num_F = readImages("tests_F")
    test_weights_T = getWeights(test_images_T, mean, eigen_vectors)
    test_weights_F = getWeights(test_images_F, mean, eigen_vectors)
    for i in range(2):
        x = []
        y = []
        co = covariance if i == 1 else None
        for k in ks:
            results_T,_ = getResults(train_weights, test_weights_T, k=k, covariance=co)
            results_F,_ = getResults(train_weights, test_weights_F, k=k, covariance=co)
            accuracy = (np.array(results_T).sum() +  num_F - np.array(results_F).sum()) / (num_T + num_F)
            x.append(k)
            y.append(accuracy)
            
        if i == 0:
            label = "euclidean"
            marker = '.'
            color = 'red'
        else:
            label = "mahalanobis"
            marker = '*'
            color = 'blue'
        plt.plot(x, y, marker=marker, color=color, label=label, linewidth=2)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
# show the performance for different C and gamma in svm
def svmStats(weights, age):
    CN = 6
    
    gamma = [0.01, 0.05, 0.1, 0.5]
    C = [ 1, 10, 50, 100, 500, 1000]
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma, 'C': C}]
#                     {'kernel': ['rbf'],
#                      'C': [ 1, 10, 50, 100, 500, 1000, 5000, 10000, 50000], 
#                      'gamma': [0.5, 0.1, 0.05, 0.01, 0.05, 0.001 ]},
#                     {'kernel': ['linear'], 
#                      'C': [ 1, 10, 50, 100, 500, 1000, 5000, 10000, 50000],
#                      'gamma': [0.5, 0.1, 0.05, 0.01, 0.05, 0.001 ]}

    score = 'accuracy'
    print("# Tuning hyper-parameters for %s" % score)
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                        scoring='%s' % score)
    clf.fit(weights, age)
    means = clf.cv_results_['mean_test_score']
    
    ind = np.arange(CN)
    width = 0.2
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    
    
    rects1 = ax.bar(ind, means[:6], width, color='r')
    rects2 = ax.bar(ind+width, means[6:12], width, color='b')
    rects3 = ax.bar(ind+width*2, means[12:18], width, color='g')
    rects4 = ax.bar(ind+width*3, means[18:24], width, color='y')
    
    def addText(rects, scores):
        for i  in range(len(rects)):
            rect = rects[i]
            score = scores[i]
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.01*h, '%.2f'%score,
                    ha='center', va='bottom')
        
    addText(rects1, means[:6])
    addText(rects2, means[6:12])
    addText(rects3, means[12:18])
    addText(rects4, means[18:24])
    
    ax.set_ylabel('Scores')
    ax.set_xlabel('C')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(C)
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('gamma: 0.01', 'gamma: 0.05', 'gamma: 0.1', 'gamma: 0.5') )
    
    for mean, params in zip(means, clf.cv_results_['params']):
        print("%0.3f for %r"
              % (mean, params))
    print()
    plt.show()

# main control 
def main():
    global TEST_DIR, MULTI_FACE, AGE_TEST_DIR, middle_point
     
    print("Loading Training Images...")
    train_T, train_T_size = readImages(TRAIN_DIR_T)
    print("Image size: {}*{} pixels".format(image_size[1], image_size[0]) )
    middle_point = train_T_size 
     
    train_F,_=  readImages(TRAIN_DIR_F)
    train_images = np.append(train_T, train_F, axis=0)
    print("")
     
     
    print("Calculating PCA... ")
    mean, variance, eigen_vectors = PCA(train_T)
     
    # Covaraince for the calculating mahalanobis distance 
    train_weights_T = getWeights(train_T, mean, eigen_vectors)
    covariance = np.cov(train_weights_T.T)
     
    # Store the eigenface weight for training data
    train_weights = getWeights(train_images, mean, eigen_vectors)
    print("")  
     
    print("Loading Age Training Images...")
    age_img_train,_ = readImages(AGE_TRAIN_DIR)
    age_train_weights = getWeights(age_img_train, mean, eigen_vectors)
    age_train = readAge(AGE_TRAIN_DIR)  
    clf = SVC(kernel='rbf', C=50, gamma=0.5)
    clf.fit(age_train_weights, age_train)
    print("")   
    
    print("")
    help = '''Command options:
        show --- show the mean face and top 20 eigenfaces
        test [FOLDER_PATH] --- test the images of given folder if not provided the default will be used
        age [FOLDER_PATH] --- predict the age of the images in the given folder if not provided the default will be used 
        multi [FILE_PATH] --- detect multi-face in the provided image if not provided the default will be used
        webcam [Name]--- when the name is provided the webcam will record 50 faces of that individual.
                        when the name is not provided, a real-time face recognition is enabled with name tags.
        stats [OPTIONS] --- show the evaluation graphs with options [m, k, svm]
        help --- show this command intruction again
        quit --- quit the program'''
    print(help)
    while True:
        command = input("Please enter commands: ")
        base = command.split(" ")
        
        # show mean and eigenfaces
        if base[0] == "show":
            show(mean, eigen_vectors, variance)
            
        # Testing
        elif base[0] == "test":
            if len(base) > 2:
                print("Invalid Command length!")
                continue
            if len(base) == 2:
                TEST_DIR = base[1]
            
            test(mean, eigen_vectors, train_weights)
            
        # Age Prediction
        elif base[0] == "age":
            if len(base) > 2:
                print("Invalid Command length!")
                continue
            if len(base) == 2:
                AGE_TEST_DIR = base[1]
                
            agePrediction(clf, mean, eigen_vectors)
            
        # MultiScale Sliding Window Test
        elif base[0] == "multi":
            if len(base) > 2:
                print("Invalid Command length!")
                continue
            if len(base) == 2:
                MULTI_FACE = base[1]
            
            multiTest(mean, eigen_vectors, train_weights, covariance=covariance)
           
        # WebCam 
        elif base[0] == "webcam":
            if command == "webcam":  
                webcam()
            if len(base) == 2:
                if len(base[1]) != 0:
                    webcam_train(base[1])
                    webcam()
        elif base[0] == "help":
            print(help)
        
        # show the stats for m, k and svm
        elif base[0] == "stats":
            if len(base) != 2:
                print("Invalid Command length!")
                continue
            if base[1] == "m":
                mStats(train_T, train_images)
            elif base[1] == "k":
                kStats(mean, eigen_vectors, train_weights, covariance)
            elif base[1] == "svm":
                svmStats(age_train_weights, age_train)
        
        # quit the program
        elif base[0] == "quit":
            break
        else:
            print("Invalid Command!")
            
        plt.close('all')
        cv2.destroyAllWindows()
        print("")

if __name__ == '__main__':
    main()