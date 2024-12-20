# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
# 他の必要なライブラリをインポート

# モデルの定義
def build_model(input_dim, hidden_nodes=50, initial_lr=0.001, step_rate=10, decay=0.95):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=step_rate * (len(x_train) // BATCH_SIZE),
        decay_rate=decay,
        staircase=True
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_nodes, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(hidden_nodes, activation='relu'),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(hidden_nodes, activation='relu'),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(hidden_nodes, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

# モデルの読み込みまたは作成
    if 'model' not in st.session_state:
        model = tf.keras.Sequential(...)
        try:
            model.load_weights("model.h5")
        except OSError as e:
            print("モデルファイルが見つかりません:", e)
    return model

# ページタイトル
st.title("内部摩擦角予測アプリ")

# サイドバーに説明文などを表示
st.sidebar.title("入力値")

# 各入力値をユーザーが入力できるようにする
#depth = st.sidebar.number_input("深さ", min_value=0.0, max_value=10000.0)
#void_ratio = st.sidebar.number_input("間隙率", min_value=0.0, max_value=10000.0)
#water_content = st.sidebar.number_input("自然含水比", min_value=0.0, max_value=10000.0)
#X = st.sidebar.number_input("X", min_value=-500.0, max_value=10000.0)
#Y = st.sidebar.number_input("Y", min_value=-500.0, max_value=10000.0)
#UU = st.sidebar.radio("UU (0 or 1)", options=[0, 1])
#CU = st.sidebar.radio("CU (0 or 1)", options=[0, 1])
#CUBar = st.sidebar.radio("CUBar (0 or 1)", options=[0, 1])
#CD = st.sidebar.radio("CD (0 or 1)", options=[0, 1])

# 一般
st.sidebar.header("一般")
depth = st.sidebar.number_input("深さ", min_value=0.0, max_value=10000.0)
void_ratio = st.sidebar.number_input("間隙率", min_value=0.0, max_value=10000.0)
water_content = st.sidebar.number_input("含水比", min_value=0.0, max_value=10000.0)

# 座標（土質により決定）
st.sidebar.header("座標（土質により決定）")
X = st.sidebar.number_input("X", min_value=-500.0, max_value=10000.0)
Y = st.sidebar.number_input("Y", min_value=-500.0, max_value=10000.0)

# 三軸圧縮試験
st.sidebar.header("三軸圧縮試験")
UU = st.sidebar.radio("UU (0 or 1)", options=[0, 1])
CU = st.sidebar.radio("CU (0 or 1)", options=[0, 1])
CUBar = st.sidebar.radio("CUBar (0 or 1)", options=[0, 1])
CD = st.sidebar.radio("CD (0 or 1)", options=[0, 1])

# 入力値をNumPy配列に変換
input_data = np.array([[depth, void_ratio, water_content, X, Y, UU, CU, CUBar, CD]]).astype(np.float32)

model_path='model.h5'
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except OSError as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return None

# モデルのロード
model = load_model(model_path)

# 予測の実行
prediction = model.predict(input_data)

# 追加の説明文
st.write("**入力値のx,yについて**")
st.write("礫（-75.62，38.14）、砂（-122.47，-114.98）、シルト（30.24，-162.03）、粘土（77.65，-9.26）の組み合わせを入力してください。")

# 予測結果を表示
st.write("**予測結果:**")
st.write("入力されたパラメータに基づいて、内部摩擦角（φ）を予測しました。")
st.write("予測結果は、", prediction[0][0], "です。")
