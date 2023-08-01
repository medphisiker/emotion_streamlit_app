import io
import os
from glob import glob
import shutil

import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
from emotion_video_track import EmotionVideoTrack
import pandas as pd


def get_video_path(path2image):
    path2video = os.path.splitext(path2image)[0] + ".mp4"
    return path2video


if __name__ == "__main__":
    emotion_video_track = EmotionVideoTrack()

    # Верстка
    st.title("Распознавание эмоций участников онлайн конференции на видео")
    st.title(f"Модель использует, - {emotion_video_track.facedet.device}")
    st.markdown("Выберите видео одним из любых предложенных способов ниже.")
    st.markdown("Затем нажмите кнопку `Распознать эмоцию` внизу.")

    # блок распознавания по готовым примерам
    st.header("Способ 1: Выберите в качестве видео один из наших примеров:")

    path2image = image_select(
        label="Выберите одно из наших видео",
        images=[
            "test_video/RAVDESS_actress_24_with_interruption.jpg",
            "test_video/RAVDESS_zoom_demo.jpg",
        ],
        captions=["RAVDESS_actress_24_with_interruption", "RAVDESS_zoom_demo"],
        use_container_width=False,
    )

    path2video = get_video_path(path2image)

    recognize_our_pic = st.button("Распознать эмоцию на нашем примере")

    if recognize_our_pic:
        try:
            if os.path.exists('output'):
                shutil.rmtree('output')
            
            emotion_video_track.track_emotions_on_video(
                path2video,
                "output",
                "emotion_video_annotation.csv",
                "emotion_video_visualisation.mp4",
            )

            video_file = open("output/emotion_video_visualisation.mp4", "rb")
            video_bytes = video_file.read()

            st.text("Видео с визуализацией распознавания эмоций")
            st.video(video_bytes)

            df = pd.read_csv("output/emotion_video_annotation.csv")
            st.text("Разметка кадров видео с распознанными эмоциями")
            st.dataframe(df)

            # получаем пути к файлам
            persons_faces_paths = sorted(glob("output/persons_faces/*.jpg"))
            makdown_folder_content = ["output/persons_faces"]
            for person_face_path in persons_faces_paths:
                filename = os.path.split(person_face_path)[-1]
                filename = f"├── {filename}"
                makdown_folder_content.append(filename)

            makdown_folder_content = "\n".join(makdown_folder_content)
            print(makdown_folder_content)

            st.markdown(
                """
                Однако из таблицы не ясно, кто является персоной с поярковым номером `id` и как она выглядит.
                Поэтому скрипт сделает папку `output/persons_faces` в которой будет по одному crop'у лица для каждого участника онлайн конференции в момент его появления:               
                """
            )
            st.markdown(f"```{makdown_folder_content}```")

            for person_face_path in persons_faces_paths:
                image = Image.open(person_face_path)
                st.image(image, caption=f"{person_face_path}")

        except TypeError:
            st.write("Вы не выбрали видео")
