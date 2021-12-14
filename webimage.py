import streamlit as st  # streamlit==0.61.0
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflowjs as tfjs

from streamlit_webrtc import webrtc_streamer


webrtc_streamer(key="example")


st.title('Webカメラで遊ぼう！')

st.write('interactive Widgets')


class VideoProcessor:
    def __init__(self) -> None:
        self.threshold1 = 100
        self.threshold2 = 200
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        img = cv2.cvtColor(cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
if ctx.video_processor:
    ctx.video_processor.threshold1 = st.slider("Threshold1", min_value=0, max_value=1000, step=1, value=100)
    ctx.video_processor.threshold2 = st.slider("Threshold2", min_value=0, max_value=1000, step=1, value=200)

#classes = ({0:'angry',1:'disgust',2:'fear',3:'happy',
#          4:'sad',5:'surprise',6:'neutral'})

#image_path='/Users/kambayashishouhei/desktop/koba_a.jpg'


#img = image.load_img(image_path, grayscale=True , target_size=(64, 64))
#img_array = image.img_to_array(img)
#pImg = np.expand_dims(img_array, axis=0) / 255

#model_path = '/Users/kambayashishouhei/desktop/fer2013_mini_XCEPTION.110-0.65.hdf5'

#emotions_XCEPTION = load_model(model_path, compile=False)

#prediction = emotions_XCEPTION.predict(pImg)[0]

##convert the model into tf.js model
#save_path = '../nodejs/static/emotion_XCEPTION'
#tfjs.converters.save_keras_model(emotions_XCEPTION, save_path)
#print("[INFO] saved tf.js emotion model to disk..")

#top_indices = prediction.argsort()[-5:][::-1]
#result = [(classes[i] , prediction[i]) for i in top_indices]
#for x in result:
#    print(x)


#5/31メモ
#他のモデルを取ってくるのでもいいのでは？
#顔のどこをみているのか？
