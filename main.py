import cv2  # opencv-python==4.2.0.34
import streamlit as st  # streamlit==0.61.0
import tensorflow as tf  # tensorflow==2.2.0
from tensorflow import keras


def get_model():
    model = keras.applications.MobileNetV2(include_top=True, weights="imagenet")
    model.trainable = False
    return model


def get_decoder():
    decode_predictions = keras.applications.mobilenet_v2.decode_predictions
    return decode_predictions


def get_preprocessor():
    def func(image):
        image = tf.cast(image, tf.float32)

        image = tf.image.resize(image, (224, 224))
        image = keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)
        return image

    return func


class Classifier:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.model = get_model()
        self.decode_predictions = get_decoder()
        self.preprocessor = get_preprocessor()

    def predict(self, image):
        image = self.preprocessor(image)
        probs = self.model.predict(image)
        result = self.decode_predictions(probs, top=self.top_k)
        return result


def main():
    st.markdown("# Image Classification app using Streamlit")
    st.markdown("model = MobileNetV2")
    device = user_input = st.text_input("input your video/camera device", "0")
    if device.isnumeric():
        device = int(device)
    cap = cv2.VideoCapture(device)
    classifier = Classifier(top_k=5)
    label_names_st = st.empty()
    scores_st = st.empty()
    image_loc = st.empty()

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = classifier.predict(frame)
        labels = []
        scores = []
        for (_, label, prob) in result[0]:
            labels.append(f"{label: <16}")
            s = f"{100*prob:.2f}[%]"
            scores.append(f"{s: <16}")
        label_names_st.text(",".join(labels))
        scores_st.text(",".join(scores))
        image_loc.image(frame)
        if cv2.waitKey() & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    main()


#from typing import Text
#import streamlit as st
#import numpy as np
#import pandas as pd
#from PIL import Image
#import time

#st.title('Streamlit 超入門')

#st.write('interactive Widgets')

#latest_iteration=st.empty()
#bar=st.progress(0)

#for i in range(100):
#    latest_iteration.text(f'Iteration {i+1}')
#    bar.progress(i+1)
#    time.sleep(0.4)


#'done'


#left_column, right_column=st.beta_columns(2)

#button = left_column.button('右カラムに文字を表示')
#if button:
#    right_column.write('ここは右カラムです')

#expander1=st.expander('問い合わせ1')
#expander1.write('問い合わせ回答')
#expander1=st.expander('問い合わせ2')
#expander1.write('問い合わせ回答2')
#expander1=st.expander('問い合わせ3')
#expander1.write('問い合わせ回答3')




#text = st.text_input('あなたの趣味を教えてください')
#condition = st.slider('あなたの今の調子は？',0,100,50)

#'あなたの趣味:',text, 'です'
#'コンディション:',condition




#option = st.selectbox(
#    'あなたが好きな数字を教えてください,',
#    list(range(1,11))
#)
#'あなたの好きな数字は、', option, 'です。'

#if st.checkbox('Show image'):
#    img=Image.open('onomichi.jpg')
#    st.image(img, caption='Shohei Kambayashi', use_column_width=True)



#df = pd.DataFrame(
#    np.random.rand(100, 2)/[50,50]+[35.69,139.70],
#    columns=['lat', 'lon']
#)
#st.table(df.style.highlight_max(axis=0))

#st.map(df)





"""
# 章
## 節
### 項

```python
import streamlit as st
import numpy as np
import pandas as pd
```
"""