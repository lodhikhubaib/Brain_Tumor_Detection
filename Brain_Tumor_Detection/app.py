import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
import numpy as np

def main():
    def front_page():
        st.title("Brain Tumor Detection App")
        st.write("Welcome to the Brain Tumor Detection App!")
        if st.button("Get Started"):
            st.session_state.page = "Detection Page"
            st.experimental_rerun()
    
    def detection_page():        
        st.set_page_config(
            page_title="Brain Tumor Detector",
            page_icon="Icon\Fav_Icon_Brain_Tumor.jpeg",
            layout="wide",
        )

        labels = ['no' , 'yes']

        st.title('Brain Tumor  Detector'+ ':brain:')
        st.write('##### This App can detect Brain Tumor with 92% Accuracy.')

        st.sidebar.write("""## Please upload an Image""")

        def load_model():
            model_path = 'Brain_Tumor_tf_2.16.1\content\Brain_Tumor_tf_2.16.1'
            loaded_model = tf.saved_model.load(model_path)
            return loaded_model

        image = st.sidebar.file_uploader('', type=['jpeg', 'jpg', 'png'])

        if image is not None:
            image = Image.open(image).convert('RGB')
            st.image(image, width=480)
            
            # Resize Image to 224 by 224:
            image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
            
            # Convert image to numpy array:
            image_array = np.asarray(image)
            
            # Set model input:
            image_data = np.reshape(image_array, [1, 224, 224, 3])
            
            # Making a prediction
            model = load_model()
            infer = model.signatures["serving_default"]
            pred = infer(tf.constant(image_data, dtype=tf.float32))
            
            # Find the correct output key
            output_key = list(pred.keys())[0]
            
            index = np.argmax(pred[output_key])
            name = labels[index]
            conf = pred[output_key][0][index]
            st.write(f'### Prediction: {name}')
            st.write(f'### Confidence: {conf*100:0.2f}%')
            
    if "page" not in st.session_state:
        st.session_state.page = "Front Page"

    if st.session_state.page == "Front Page":
        front_page()
    elif st.session_state.page == "Detection Page":
        detection_page()


if __name__ == "__main__":
    main()
           