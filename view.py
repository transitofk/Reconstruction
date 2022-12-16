import streamlit as st
import cv2
import numpy as np
from constructing_a_super_resolution import tests, reconstruction, reconstruction_primitive, choose_filter


def main_page():
    # img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    global img, center, w, h, img_HR_cur
    shift_x_y = []
    rot = []
    list_min = []

    st.title("Реконструкция  ЧБ-изображений")
    uploaded_file = st.file_uploader("Загрузить фото", type=['jpg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 0)
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        img_HR_cur = np.zeros((h, w), np.uint8)
        st.write("Оригинал")
        st.image(img)

    with st.sidebar:
        type_of_rec = st.sidebar.selectbox('Реконструкция',
                                           ('Итеративная', 'Примитивная'))
        if type_of_rec == 'Итеративная':
            num_rec = 1
        else:
            num_rec = 2
        num_frame = st.number_input("Количесво кадров", min_value=1, value=1)
        size_kernel = st.number_input("Размер ядра", min_value=1, value=1)
        if size_kernel % 2 == 0: st.text("Размер ядра должен быть\nнечетным числом")
        filter = st.sidebar.selectbox('Фильтр',
                                      ('Равномерное размытие', 'Гауссово размытие', 'Двустороннее размытие'))
        if filter == 'Равномерное':
            num_filter = 1
        elif filter == 'Гауссово':
            num_filter = 2
        else:
            num_filter = 3

        if num_rec == 1:
            num_cycle = st.number_input("Количесво циклов", min_value=1, value=1)
            alpha = st.number_input("Параметр alpha", min_value=0.001, value=0.1, step=0.05)

        but_start = st.button('Сгенерировать')

    if but_start:
        tab1, tab2 = st.tabs(["Результат", "Синтетические тесты"])
        with tab1:
            for i in range(num_frame):
                shift_x_y.append(np.random.randint(1, 15))
                rot.append(np.random.randint(-15, -1))

            if size_kernel % 2 == 1 and num_rec == 1:
                list_min = tests(num_frame, list_min, img, rot, shift_x_y, center, w, h, size_kernel, num_filter)
                res = reconstruction(num_cycle, img_HR_cur, list_min, rot, shift_x_y, alpha, center, w, h, size_kernel,
                                     num_filter)
                st.image(res)
            if size_kernel % 2 == 1 and num_rec == 2:
                list_min = tests(num_frame, list_min, img, rot, shift_x_y, center, w, h, size_kernel, num_filter)
                res = reconstruction_primitive(img_HR_cur, list_min, rot, shift_x_y, center, w, h)
                st.image(res)
            with tab2:
                for index, elem in enumerate(list_min, start=1):
                    st.write('Кадр ', index, ':')
                    st.image(elem)
