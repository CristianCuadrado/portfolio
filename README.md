# MlOps and Computer Vision Engineer
MLOps Engineer specializing in image processing, with experience in audio and text domains. Knowledgeable in Cloud technologies (AWS and Azure) to bolster professional capabilities.

With 8 years in artificial intelligence, I consider myself a knowledgeable professional deeply interested in its constant evolution.

In computer vision, my expertise spans modern neural network techniques and traditional algorithm-based approaches. Whether addressing text or audio challenges, I am a competent worker who has developed various solutions.

Motivated and adaptable, I thrive in diverse project environments, having worked in both startups and large corporations with varying contexts. Capable of swiftly producing functional prototypes, even in data-scarce scenarios, or refining systems at the cutting edge of the state of the art.

## Studies
Master in the UPC (MATT) specialized in Deep Learning applied to image, audio, text and data.

## Experience as developer
- 9 years of experience in Machine Learning.
- 7 years of experience in Computer Vision. Strong knowledge of OpenCV.
- 6 years of experience in Deep Learning. Experience with Pytorch and Keras frame-
works. Experience with algorithms such as Image Classification, Object Detection,
Image Segmentation, CycleGan, Reinforcement Learning (DDPG), Generative algo-
rithms, Encoder-Decoder and Recurrent Neural Networks.
- 2 years of experience in research.
- 1 year experience with Generative AI for image. I have experience with different
algorithms such as text-to-image, image-to-image, IP-Adapter, inpainting, Control-
Net, and LoRAs.
- Experience in Natural Language Processing and Audio Processing.
- Experience with Python, Matlab and R.
- Experience with AWS, Google Cloud, Azure and Replicate.
- Experience with SQL, NoSQL databases and Docker.

## Interpersonal Skills
- Experience in dealing with clients. 
- Experience managing teams.
- Experience in international projects.

## Languages 
Spanish, Catalan, English.

## My sites
- My linkedin page: https://www.linkedin.com/in/cristian-cuadrado-conde/
- Sometimes I write articles in Medium. Please check it out https://medium.com/@c.cuadrado91

## My projects
This comprises a compilation of select projects I have executed across diverse enterprises.
### Characterization
<p align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/viking.png" width="200">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/linkedin.png" width="200">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/american.png" width="200">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/old_money.png" width="200">
</p>
This project was for a mobile app. It involved creating avatars from a user's photo and a reference image, using Generative AI algorithms

#### Technologies
- Python as main language.
- Huggingface for generative models.
- Replicate for the infrastructure.
  
### Image Similarity in fashion industry
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/img_comparison.jpg?raw=true" width="500">
</div>
This project was undertaken for a fashion industry client. The objective was to enable users to compare a piece of clothing with the entire catalog. Initially, I employed an image comparison algorithm, extracting a feature vector that could be utilized for comparing items. Subsequently, I enhanced it with classification algorithms to identify garment characteristics, filter potential errors, and improve performance by discarding numerous irrelevant items.

#### Technologies
- Python as main language.
- Pytorch for the image similarity and classification algorithms.
- Azure for the infrastructure.

### Tagging for fashion industry
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/tag_fashion.jpg?raw=true" width="500">
</div>

This project was undertaken for a fashion industry client. The objective was to create an algorithm capable of automatically adding tags to the catalog each time a new garment was added. These tags were supposed to encompass different characteristics such as the garment category (pants, coats, t-shirts...), the subcategory (shorts, short-sleeve shirt...), specific features depending on the type (round neck, high neck), the material (cotton, leather, silk), and others. To achieve this, I used different image classification algorithms, each trained in a specific type.

#### Technologies
- Python as main language.
- Pytorch for the classification algorithms.
  
### Object Segmentation in Underground images
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/underground.jpg?raw=true" width="500">
</div>

This project was my final master's thesis. Its aim was to identify marine fauna in their natural environment for biodiversity assessment using video recordings. The project posed particular challenges as the camera was movable, the underwater background was dynamic, and the fish we aimed to detect were specialized in camouflage in that region. Additionally, the availability of labeled data was very limited. 

Four different algorithms were employed to attempt fish identification. The ones yielding the best results were an image segmentation algorithm combined with a video flow detection algorithm. Other tests were conducted, though with less success, using object detection algorithms and anomaly detection in images.

Nevertheless, the project succeeded in providing provisional labeling of the images, aiding biologists in their analysis. The project assumed that with more labeled data, it would be possible to develop an algorithm with human-level performance.

You can find the complete work at the following link: https://upcommons.upc.edu/handle/2117/334349.

#### Technologies
- Python as main language.
- Pytorch for the image segmentation, flow detection, image anomaly and object detection algorithms.
  
### Object Segmentation for medical image
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/segmentation.jpg?raw=true" width="200">
</div>
This project involved using an object segmenter to aid in the clinical study of wound treatment of various kinds. We needed to be able to detect the area affected by a wound, identify the type of tissue, define its proportion and area, so that from various shots over time, the progression of the wound could be evaluated. In this project, an object segmenter was used to define the segmentation task, along with other algorithms to assess the wound size, and a system based on superpixels for labeling the training database.

#### Technologies
- Python as main language.
- OpenCV for super pixels algoritms. 
- Pytorch for the image segmentation algorithm.
  
### Devices identification 
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/devices_identification.jpg?raw=true" width="200">
</div>
This project was a component of a solution aimed at ensuring students did not engage in cheating during online exams using the webcam. It aimed to identify whether the student was using unauthorized items such as headphones or a mobile phone. This was achieved with image classification algorithms.

#### Technologies
- Python as main language.
- Pytorch for the Classification algorithm.
- AWS for the infrastructure.

### Light correction
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/gamma_correction.jpg?raw=true" width="500">
</div>
This project was a component of a solution aimed at ensuring students did not engage in cheating during online exams using the webcam. It was necessary to create a module to apply light corrections to improve some images. This was done using traditional image processing algorithms, in this case, employing dynamic gamma correction. The algorithm was fully implemented using numpy and OpenCV libraries.

#### Technologies
- Python as main language.
- OpenCV for algorithm implementation.
- AWS for the infrastructure.

### Voice to text
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/voice2text.png?raw=true" width="500">
</div>
This project was a component of a solution aimed at ensuring students did not engage in cheating during online exams using the microphone. This module was designed to detect voices, ensuring that in the event of someone speaking, it would transcribe the speech for analysis to determine if the student was receiving assistance or information. The system operated by detecting sounds of a certain intensity; if it exceeded the permissible threshold, an algorithm was employed to identify whether the sound was human speech. If affirmative, the audio was transcribed into text for subsequent analysis.

#### Technologies
- Python as main language.
- SpeechRecognition for Speech Recognition and Voice to Text. 
- AWS for the infrastructure.

### Face Identification
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/face_identification.png?raw=true" width="500">
</div>
This project was a component of a solution aimed at ensuring students did not engage in cheating during online exams using the webcam. This module was responsible for verifying that the student was the same person who had registered and that no one was impersonating them. To achieve this, I implemented a facial identification system based on ArcFace, which was the state of the art at that time. I enhanced this system by training it with the client's data to make the model more robust against issues associated with webcams, such as lighting, angle, and image quality. Emphasizing that facial identification algorithms operate using two components: one for Face Detection and another for Facial Comparison.

#### Technologies
- Python as main language.
- Pytorch for the Face Detection and Face Comparison algorithms.
- AWS for the infrastructure.

### People tracking
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/people_traking.png?raw=true" width="500">
</div>
This project was carried out in an industrial environment. In this case, the goal was to roughly assess how many people were in each room simultaneously. Therefore, it was important to be able to identify a person in their movement and not confuse them with others, as well as to identify the region of the image where the doors were located to determine if someone had exited or entered. To achieve this, object detectors were used to locate individuals, tracking algorithms, and motion estimation algorithms.

#### Technologies
- Python as main language.
- Pytorch for the Object Detection(YOLO 5) and Motion Estimation(PWC-Net) algorithms.

### Anomaly Detection for temporal series
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/anomaly_detection.png?raw=true" width="300">
</div>

I have implemented this project in various time series contexts, with notable applications including anomaly detection in motors through their sensors and assessing measurements taken by different devices in medical analysis. These algorithms aim to identify signals that deviate from the norm established during training data analysis. They prove highly effective in detecting operational failures.

#### Technologies
- Python as main language.
- Scikit-learn for implementation.
- Some examples of the implemented algorithms are: Isolation Forest, One-Class SVM, k-Nearest Neighbors and Z-score.

### OCR and Natural Language Processing

#### Denoising
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/denoise.jpg?raw=true" width="300">
</div>

#### Entity extraction
<div align="center">
  <img src="https://github.com/CristianCuadrado/portfolio/blob/main/images/entity_extraction.jpg?raw=true" width="300">
</div>

In this project, the client requested a market study based on restaurant menus downloaded from the internet. To achieve this, I employed Optical Character Recognition (OCR) to extract the text, alongside a cleaning step to remove undesirable elements and an image classification algorithm to identify the menu format (one, two or more columns). 
After extracting the text, I applied Natural Language Processing algorithms to extract the information relevant to their interests.

#### Technologies
- Python as main language.
- OpenCV for cleaning. 
- Pytorch for the classification algorithm.
- Tesseract as OCR.  
- Stanza as entity extraction. 
- Azure for the infrastructure. 
