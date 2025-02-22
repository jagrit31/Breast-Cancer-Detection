import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.weights.h5")
    return loaded_model

def predict_image(model, img_array):
    prediction = model.predict(img_array)
    return prediction

def main():
    st.title('Breast Cancer Detection')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(25, 25))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = img_array.reshape(1, 25, 25, 3)  # Reshape for model input

        if st.button('Predict'):
            model = load_model()
            prediction = predict_image(model, img_array)

            st.write('Prediction:')
            st.write('Cancer: {:.2f}%'.format(prediction[0][0] * 100))
            st.write('No Cancer: {:.2f}%'.format(prediction[0][1] * 100))

            # Display prediction scores as a bar chart
            labels = ['Cancer', 'No Cancer']
            scores = [prediction[0][0], prediction[0][1]]
            fig, ax = plt.subplots()
            ax.bar(labels, scores)
            ax.set_ylabel('Prediction Score')
            ax.set_title('Prediction Scores')
            st.pyplot(fig)

            final_prediction = 'Cancer' if prediction[0][0] > prediction[0][1] else 'No Cancer'
            st.write('Final Prediction:', final_prediction)


if __name__ == '__main__':
    main()
