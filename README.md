To run the realtime version, download the *Gender Classification - Webcam.py* file and also download the pre-trained models from here : (https://drive.google.com/drive/u/0/folders/1EEHsufmsvIwDVhTTouNYXfkhGb7UKKVq)

Place the *Gender Classification - Webcam.py* and the 4 files in the same directory.

Run the python file, an instance will open accessing your webcam, displaying the gender along with the face region highlighted on the display.

-------------------------------------

**Realtime Gender Classification using Pre-trained models**

**_Author:_ Gowtham M**

**Introduction:**

In this project, we aim to classify the gender of individuals based on their facial features using pre-trained models, in this context, binary classification (male/female)

**Problem Statement:**

- The primary goal of this project is to develop a gender classification system capable of accurately predicting the gender of individuals from facial images.
- Challenges include face detection in varying lighting conditions, poses, as well as tackling biases in gender prediction.

**Scope:**

1. Implement face detection using a pre-trained model to detect faces in input images.
2. Use a pre-trained gender classification model to predict the gender of detected faces.
3. Evaluate the performance of the gender classification system in terms of confidence score.

**Dependencies:**

1. OpenCV library (for image preprocessing, face detection tasks)
2. **Pre-trained Models:**
    1. For face detection: "opencv_face_detector.pbtxt" and "opencv_face_detector_uint8.pb"
    2. For gender classification: "gender_deploy.prototxt" and "gender_net.caffemodel"
3. **Dataset:** The Adience dataset, containing images suitable for age and gender estimation, is used to train the pre-trained gender classification model.
4. **Download the required models:**
    1. Caffe model for [gender classification](https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ) and [deploy prototext](https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP).
    2. Face detection required files : [Gender-Prediction](https://drive.google.com/drive/folders/1i6rSVDC1XHqHru4GvIA1WrdFG22NugBc?usp=sharing)

**About the dataset:**

Since we used pretrained models, we haven’t trained the model using any dataset during runtime, the pre-trained model uses the Adience dataset, which contains images that represent some of the challenges of age and gender estimation from real-world, unconstrained images. Most notably, extreme blur (low-resolution), occlusions, out-of-plane pose variations, expressions and more.. It has a total of 26,580 photos of 2,284 subjects in eight age range.

**Methodology:**

- Data Collection: Since it’s a pretrained model, we just pre-process our input data
- Data Preprocessing: Normalize input images and preprocess them for compatibility with pre-trained models.
- Face Detection (Real-time): Implement face detection in real-time using OpenCV and a webcam feed.
- Face Detection (Static Images): Implement face detection on static images using OpenCV.
- Gender Classification: Utilize pre-trained gender classification models to predict the gender of detected faces.
- Evaluation: Evaluate the performance of the system in terms of confidence score, and real-time processing efficiency.

**To visualize how this program works (HIGH LEVEL OVERVIEW) :**

1. Input:

An input image containing 1 or more face

1. Detect Faces:
    - To detect faces in the image, a pre-trained model (face detection model) is used.
    - We will draw bounding boxes (in simple, a rectangular box) around the face, which is detected by the model.
2. Bounding box extraction :

We will get the coordinates of the bounding boxes for the detected face (x1, x2, y1, y2) (i.e. the 4 corners of the box)

1. Gender Classification :
    1. Crop the detected face part and put this as input into the gender prediction pre-trained model. As we know the coordinates, we will just get the face part and ignore the rest of the details in the image, as we will use face to classify the gender.
    2. We'll have a set of labels: Male/Female - the pre-trained model will predict whether the face resembles male or female, and the predicted one will be displayed, along with a confidence score (how strong the prediction is, as the model thinks.

5\. Output:

1. Bounding box is drawn on the detected face of the input image.
2. Gender & Confidence Score labels are shown along with it.

**How the program works:**

1. Import the OpenCV library
2. Get the input image (path)
3. To find the gender of the person in the image, we need to first detect the face regions, as the model we’re using is trained on Adience dataset, which will be suitable for our binary classification (Male/Female).
4. Get the face region in the image using a pre-trained face detection model.
    1. **Face Detection Function:**
        1. This function takes the face detection model, input image and the confidence threshold value as parameters. The confidence threshold is used later to only allow detected faces above this value.
        2. We take a copy of the original image to avoid any modifications to the input.
        3. Get the input image height and width.
            1. Shape\[0\] represents rows
            2. Shape\[1\] represents columns
        4. Before feeding this input image into the pre-trained model, we need to pre-process the image by normalizing, scaling, mean subtraction and other factors. Here, we don't perform any scaling (default=1), a common size, The values (104, 117, 123) represent the mean pixel intensities used for mean subtraction during image preprocessing, aiming to center pixel values around zero for improved model performance.
        5. Now, the blob, which is a pre-processed image, is fed into the model and a set forward to reach the output layer.
        6. Create a empty bounding boxes list which will hold co-ordinates of the detected face region, that we’ll get
        7. The forward() gives us a multidimensional array, which has following parameters:
            1. Batch Size (0)
            2. Class (0)
            3. No.of. Detections (i)
            4. Confidence Score (2)
        8. As an image can have multiple faces, it will have multiple confidence scores, as other parameters remain same here due to the fact that we only use one image, which is a single class, and the confidence score is located in the 2-nd index of this multidimensional array.
        9. So, we’ll calculate the detection in each image till the total no.of.detections we got, i - no.of.detections
        10. The loop will calculate the confidence scores for all the detections it has made.
    2. **Real-time Gender Detection:**

- In a real-time scenario, the above steps can be executed continuously on video frames captured from a camera feed.
- Each frame from the video feed is processed using the face detection function to detect faces and extract face regions.
- The extracted face regions are then passed through the gender detection model to classify the gender in real-time.
- The predicted gender along with the corresponding bounding boxes can be overlaid on the video frames to provide real-time gender classification results.

1. Allow only the detections which have more than 0.7 confidence score, get the coordinates of the detected face region.
    1. The index 3,4,5,6, represents x1 and y1, x2 and y2 values.
    2. Then we will assign these values the empty bounding boxes list.
    3. Now, draw a box around the face detected region, with the coordinates
    4. Return the coordinates and the input image with a box around the face region.
2. Load the face detection model, then only the above steps will work right. It has 2 files.
    1. \`faceProto\` (Protobuf Text File): Contains the architecture or graph definition of the face detection model in human-readable text format.
    2. \`faceModel\` (Protobuf Binary File): Contains the trained weights and parameters of the face detection model in binary format.
3. Load the gender detection model, we’ll check if there are any co-ordinates returned from the getFace() function, if not, tell no face is detected orelse, for each of the bounding box in the image, slice the regions which only have face from the input image, set the input to the model.

**NOTE**: Earlier we pre-process the image for face detections. Now we do the gender classification model. For this model, we input the face detected image to this for further processing.

1. Now, after the model runs, it returns a vector(list) of 2 elements which shows the value of male followed by female. We will find the maximum value to display that as the predicted gender.
2. Display the detected face image with the predicted gender as the output.

**Future Work:  
<br/>**1\. Implementing age estimation

2\. Using facial features to find the emotions shown

2\. Adding more ethical considerations

**Links:**

**Realtime Gender Classification:** [Realtime Gender Classification](https://drive.google.com/drive/folders/1EEHsufmsvIwDVhTTouNYXfkhGb7UKKVq?usp=drive_link)

**To access the static image version in google colab:  
**[(COLAB) Gender Classification using Pretrained Model (Caffe Model).ipynb](https://colab.research.google.com/drive/1jet5mj1B81AkCibwDI4sWZMVhxIdHgfb?usp=sharing)
