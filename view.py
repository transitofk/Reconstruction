import streamlit as st
import cv2
import numpy as np
from constructing_a_super_resolution import tests, reconstruction, reconstruction_primitive, choose_filter


def main_page():
    global img, center, w, h, img_HR_cur, color
    shift_x_y = []
    rot = []
    list_min = []

    st.set_page_config(page_title="Reconstruction")
    st.title("Реконструкция изображений HR из набора изображений LR")
    uploaded_file = st.file_uploader("Загрузить фото:", type=['jpg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        color = cv2.split(img)  # color = [b, g, r]
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        img_HR_cur = np.zeros((h, w), np.float64)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.write("Оригинал")
        st.image(img)

    with st.sidebar:
        st.sidebar.title("Входные параметры")
        type_of_rec = st.sidebar.selectbox('Реконструкция',
                                           ('Итеративная', 'Примитивная'))
        if type_of_rec == 'Итеративная':
            num_rec = 1
        else:
            num_rec = 2

        num_frame = st.number_input("Количесво кадров", min_value=1, value=1)
        r = st.number_input("Max cмещение", min_value=1, max_value=50, value=15, step=1)
        phi = st.number_input("Max угол поворота", min_value=1, max_value=50, value=10, step=1)
        size_kernel = st.number_input("Размер ядра", min_value=1, value=1)
        if size_kernel % 2 == 0: st.text("Размер ядра должен быть\nнечетным числом")
        filter = st.sidebar.selectbox('Фильтр',
                                      ('Равномерное размытие', 'Гауссово размытие', 'Треугольное размытие'))


        if filter == 'Равномерное':
            num_filter = 1
        elif filter == 'Гауссово':
            num_filter = 2
        else:
            num_filter = 3

        if num_rec == 1:
            num_cycle = st.number_input("Количесво циклов", min_value=1, value=1)
            alpha = st.number_input("Параметр сходимости", min_value=0.001, max_value=1., value=0.1, step=0.05, format="%.5f")
            beta = st.number_input("Параметр размытия", min_value=0.00001, max_value=1., value=0.001, step=0.001,  format="%.5f")

        if num_rec == 2:
            iterpol = st.sidebar.selectbox('Интерполяция',
                                           ('Метод ближайшего соседа', 'Билинейная',
                                            'С использованием кубического B-сплайна', 'Бикубическая', 'Ланцоша'))
            iterpol_list = {
                'Метод ближайшего соседа': cv2.INTER_NEAREST,
                'Билинейная': cv2.INTER_LINEAR,
                'С использованием кубического B-сплайна': cv2.INTER_AREA,
                'Бикубическая': cv2.INTER_CUBIC,
                'Ланцоша': cv2.INTER_LANCZOS4
            }
            interp = iterpol_list.get(iterpol, "Invalid count blur")

        but_start = st.button('Сгенерировать')

    if but_start:
        tab1, tab2 = st.tabs(["Входные изображения", "Результат восстановления"])
        with tab2:
            for i in range(num_frame):
                shift_x_y.append(np.random.randint(1, r))
                rot.append(np.random.randint(-phi, phi))

            if size_kernel % 2 == 1 and num_rec == 1:
                res_color = []
                synthetic_tests_b = []
                synthetic_tests_g = []
                synthetic_tests_r = []
                for i, cur_img_color in enumerate(color):
                    list_min = tests(num_frame, list_min, cur_img_color, rot, shift_x_y, center, w, h, size_kernel,
                                     num_filter)
                    res = reconstruction(num_cycle, img_HR_cur, list_min, rot, shift_x_y, alpha, beta, center, w, h,
                                         size_kernel,
                                         num_filter)
                    res_color.append(res)
                    if i == 0:
                        synthetic_tests_b = list_min
                    elif i == 1:
                        synthetic_tests_g = list_min
                    else:
                        synthetic_tests_r = list_min
                    list_min = []
                    img_HR_cur = np.zeros((h, w), np.float64)

                imgMerged = cv2.merge(res_color)
                imgMerged = cv2.cvtColor(imgMerged, cv2.COLOR_RGB2BGR)
                st.image(imgMerged)

            if size_kernel % 2 == 1 and num_rec == 2:
                res_color = []
                synthetic_tests_b = []
                synthetic_tests_g = []
                synthetic_tests_r = []
                for i, cur_img_color in enumerate(color):
                    list_min = tests(num_frame, list_min, cur_img_color, rot, shift_x_y, center, w, h, size_kernel,
                                     num_filter)
                    res = reconstruction_primitive(img_HR_cur, list_min, rot, shift_x_y, center, w, h, interp)

                    res_color.append(res)
                    if i == 0:
                        synthetic_tests_b = list_min
                    elif i == 1:
                        synthetic_tests_g = list_min
                    else:
                        synthetic_tests_r = list_min
                    list_min = []
                    img_HR_cur = np.zeros((h, w), np.float64)

                imgMerged = cv2.merge(res_color)
                imgMerged = cv2.cvtColor(imgMerged.astype(np.uint8), cv2.COLOR_RGB2BGR)
                st.image(imgMerged)

            with tab1:
                for i, (frame_b, frame_g, frame_r) in enumerate(
                        zip(synthetic_tests_b, synthetic_tests_g, synthetic_tests_r), start=1):
                    st.write('Кадр ', i, ':')
                    imgMerged = cv2.merge([frame_b, frame_g, frame_r])
                    cv2.imwrite('temporary.jpg', imgMerged)
                    st.image('temporary.jpg')
