import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Загрузка файла .csv
st.title("Линейная Регрессия и Графики")

uploaded_file = st.file_uploader("Загрузите файл .csv", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0)

    # Визуализация данных
    st.subheader("Первые 5 строк из файла:")
    st.write(data.head())

    # Выбор фичей для регрессии
    st.subheader("Выберите фичи для регрессии:")
    features = st.multiselect("Выберите фичи:", data.columns)
    target = st.selectbox("Выберите целевую переменную:", data.columns)

    X = data[features]
    y = data[target]
    
    # Нормализация данных
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Обучение линейной регрессии
    model = LinearRegression()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)
    model.fit(X_train, y_train)

    st.subheader("Результаты регрессии:")
    results = dict(zip(features, model.coef_))
    results["intercept"] = model.intercept_
    st.write(results)

    # st.set_option('deprecation.showPyplotGlobalUse', False)  # Отключение предупреждения
    # Визуализация графиков
    st.subheader("Графики:")
    plot_type = st.selectbox("Выберите тип графика:", ["Scatter Plot", "Bar Plot", "Line Plot"])

    if plot_type == "Scatter Plot":
        st.subheader("Scatter Plot:")
        x_axis = st.selectbox("Выберите ось X:", features)
        plt.scatter(data[x_axis], data[target])
        plt.xlabel(x_axis)
        plt.ylabel(target)
        plt.xticks(rotation=45)
        st.pyplot()

    elif plot_type == "Bar Plot":
        st.subheader("Bar Plot:")
        bar_feature = st.selectbox("Выберите фичу для оси X:", features)
        plt.bar(data[bar_feature], data[target])
        plt.xlabel(bar_feature)
        plt.ylabel(target)
        plt.xticks(rotation=45)
        st.pyplot()

    elif plot_type == "Line Plot":
        st.subheader("Line Plot:")
        x_axis = st.selectbox("Выберите фичу для оси X:", features)

        sorted_data = data.sort_values(by=x_axis)
        plt.plot(sorted_data[x_axis], sorted_data[target])
        plt.xlabel(x_axis)
        plt.ylabel(target)
        plt.xticks(rotation=45)
        st.pyplot()
