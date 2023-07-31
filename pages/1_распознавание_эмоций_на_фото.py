import io

import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
from emotion_net_infer import EmotionSeqClassifier
from emotion_det_net_infer import DetectEmotionOnFaces


def load_image_from_user():
    uploaded_file = st.file_uploader(label="Выберите изображение для распознавания")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def get_webcam_photo_from_user():
    uploaded_file = st.camera_input("")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def load_image(path2image):
    image = Image.open(path2image)
    image.load()
    return image


if __name__ == "__main__":
    # инициализация модели
    # модель должна загружаться из MLFLow (он под VPN).
    # в качестве альтернативы модель грузиться из Google Drive
    emo_seq_classifier = EmotionSeqClassifier()
    emotion_det_net = DetectEmotionOnFaces(emo_seq_classifier)
    
    # Верстка
    st.title("Распознавание эмоций человека по фото")
    st.title(f"Модель использует, - {emotion_det_net.device}")
    st.markdown("Выберите изображение одним из любых предложенных способов ниже.")
    st.markdown("Затем нажмите кнопку `Распознать эмоцию` внизу.")

    # блок распознавания по готовым примерам
    st.header("Способ 1: Выберите в качестве изображений один из наших примеров:")

    path2image = image_select(
        label="Выберите одну из наших фото",
        images=[
            "example_images/angry_frame_13.jpg",
            "example_images/fearful_frame_17.jpg",
            "example_images/happy_frame_20.jpg",
            "example_images/sad_frame_47.jpg",
            "example_images/four_faces.jpg",
        ],
        captions=["angry", "fearful", "happy", "sad", "4 faces"],
        use_container_width=False,
    )

    image = load_image(path2image)

    recognize_our_pic = st.button("Распознать эмоцию на нашем примере")

    if recognize_our_pic:
        try:
            detections = emotion_det_net.detect_faces(image)
            image = emotion_det_net.viz_emo_detections(image, detections)
            st.image(image, caption="Результаты распознавания эмоций")

        except TypeError:
            st.write("Вы не выбрали фото")

    # блок распознавания по фото пользователя
    st.header("Способ 2: Загрузите свое фото с вашего компьютера")
    image = load_image_from_user()
    recognize_user_pic = st.button("Распознать эмоцию на вашем фото")

    if recognize_user_pic:
        try:
            detections = emotion_det_net.detect_faces(image)
            image = emotion_det_net.viz_emo_detections(image, detections)
            st.image(image, caption="Результаты распознавания эмоций")

        except TypeError:
            st.write("Вы не выбрали фото")

    # блок распознавания по фото с web-камеры пользователя
    st.header("Способ 3: Распознаем эмоции на вашем фото с вашей веб-камеры")

    st.markdown(
        "Чтобы сделать снимок с вашей веб-камеры, нажмите кнопку ниже `[Take Photo]`"
    )
    image = get_webcam_photo_from_user()
    recognize_user_webcam_photo = st.button("Распознать эмоцию на фото c веб-камеры")

    if recognize_user_webcam_photo:
        try:
            detections = emotion_det_net.detect_faces(image)
            image = emotion_det_net.viz_emo_detections(image, detections)
            st.image(image, caption="Результаты распознавания эмоций")

        except TypeError:
            st.write("Не удалось получить снимок с вашей веб камеры.")
            st.markdown(
                f"Для предоставления доступа к вашей веб-камере пожалуйста, "
                f"выполните инструкции приведенные [по ссылке]"
                f"(https://docs.streamlit.io/knowledge-base/using-streamlit/enable-camera)."
            )
