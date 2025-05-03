FYP Project: Visual Symphony: AI-Powered Creation of Customized Music from User Images 

This is my FYP Project, use AI analysis image's feature and predict the dominant emotion of the image user given and generate a corresponding music align the detected emotion by using customized algorithm.

The image analysis is divided to two type:
1) First is Facial image detection, the swin_base_patch4_window7_224 is a base-sized Swin Transformer model, pre-trained on ImageNet, with a 224x224 input resolution, It is trained
by using facial image dataset.
2) Second is Non Facial Image detection, EfficientNet_B2 is used as a pre-trained model on ImageNet, which helps it learn general image features, it is used to predict non-facial image's emotion.

Based on detected emotion, the algorithm will use this to generate a corresponding music track, this algorithm include:
1) Rhythm Generation Function
2) Chord Seclection
3) Melody Note Selection

User can try my app by using https://image2music.streamlit.app/.
