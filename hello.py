import streamlit as st

st.set_page_config(
    page_title="Приветствие",
    page_icon="😊",
)

st.write("# Сервис для распознавания эмоций 😊")

st.markdown(
    """
    Мы предоставляем функцию для распознавания эмоций человека по фото и видео как сервис, - function-as-a-service (FaaS).
    
    **(Это учебный некоммерческий проект 😊 мимикрирующий под стартап)**.
    
    Сервис использует State-of-Art (SOTA) подход для распознавания эмоций человека, представленный в научной статье, - 
    "Self-attention fusion for audiovisual emotion recognition with incomplete data"
    ([ссылка на статью](https://arxiv.org/abs/2201.11095), [ссылка на github](https://github.com/katerynaCh/multimodal-emotion-recognition)).
    
    **👈 Выберите страницу в левой панели навигации**, чтобы увидеть примеры
    * распознавания эмоций на фото
    * распознавания эмоций на видео
    
"""
)
