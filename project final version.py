#!/usr/bin/env python
# coding: utf-8

# In[1]:


# حذف أي تعريف قديم لدالة time_stretch من الذاكرة
try:
    del time_stretch
except:
    pass


# In[15]:


import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling1D

# import tkinter as tk
# from tkinter import filedialog
# from pydub import AudioSegment
# import sounddevice as sd
# from PIL import Image, ImageTk


# In[ ]:


get_ipython().system('pip install resampy')


# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install resampy')


# In[5]:


def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def apply_time_stretch(y, rate=0.9):
    import librosa
    try:
        return librosa.effects.time_stretch(y, rate)
    except Exception as e:
        print("⚠️ Skipping time-stretch due to error:", e)
        return y

def pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


def extract_features(file_path, augment=False):
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    features_list = []

    audios = [audio]
    if augment:
        audios.append(add_noise(audio))
        try:
            stretched = librosa.effects.time_stretch(audio, rate=0.9)
            audios.append(stretched)
        except Exception as e:
            print("⚠️ Skipping time-stretch:", e)
        audios.append(pitch_shift(audio, sr, n_steps=2))

    for a in audios:
        if len(a) < 512:
            continue

        # تحسين الخصائص الصوتية
        mfcc = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        stft = np.abs(librosa.stft(a))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=a, sr=sr)
        centroid = librosa.feature.spectral_centroid(y=a, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(a)
        rmse = librosa.feature.rms(y=a)

        # دمج كل الخصائص
        combined = np.concatenate((mfcc, delta, delta2, chroma, contrast, rolloff, centroid, zcr, rmse), axis=0)
        features = np.mean(combined.T, axis=0)
        features_list.append(features)

    return features_list


# Path to dataset directory
base_dir = r"C:\Users\HP\Downloads\TESS Toronto emotional speech set data"
data = []
labels = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_path in glob.glob(os.path.join(folder_path, '*.wav')):
            features_augmented = extract_features(file_path, augment=True)
            for f in features_augmented:
                data.append(f)
                labels.append(folder_name.split('_')[1])

# Convert data and labels to numpy arrays for model training
data = np.array(data)
labels = np.array(labels)
print(f"Extracted {len(data)} samples with {len(labels)} labels.")


# In[23]:


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


# In[37]:


from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dropout, Dense, Input, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

# تأكد أن بيانات التدريب والتجربة ثلاثية الأبعاد
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 1. Define the model (CNN + BiLSTM + Attention)
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# CNN layers for feature extraction
x = Conv1D(64, kernel_size=3, activation='relu')(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = Conv1D(128, kernel_size=3, activation='relu')(x)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.5)(x)

# BiLSTM layer for sequence processing
x = Bidirectional(LSTM(128, return_sequences=True))(x)

# Multi-Head Attention block
attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = LayerNormalization(epsilon=1e-6)(x + attention_output)

# Global Average Pooling
x = GlobalAveragePooling1D()(x)
x = Dropout(0.5)(x)

# Output layer
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(x)

# Define and compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# ----------------------------
# 2. Training with EarlyStopping and ReduceLROnPlateau
# ----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    X_train, tf.keras.utils.to_categorical(y_train),
    epochs=50,
    batch_size=32,
    validation_data=(X_test, tf.keras.utils.to_categorical(y_test)),
    callbacks=[early_stop, reduce_lr]
)

# ----------------------------
# 3. Evaluation
# ----------------------------
loss, accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test))
print('Test loss:', loss)
print('Test accuracy:', accuracy)


# In[18]:


# 1. تنبؤ النموذج على بيانات الاختبار
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


# In[39]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. استخدام النموذج للتنبؤ
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # تحويل التنبؤات إلى أرقام التصنيفات

# 2. مصفوفة الارتباك (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# 3. تقرير التصنيف (Classification Report)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print(report)


# In[ ]:





# In[ ]:


def predict_emotion(audio_file):
    features = extract_features(audio_file)
    features = features[np.newaxis, np.newaxis, :]
    print("Features shape:", features.shape)
    print("Features:", features)

    predicted_probabilities = model.predict(features)
    print("Predicted probabilities shape:", predicted_probabilities.shape)
    print("Predicted probabilities:", predicted_probabilities)

    predicted_label_index = np.argmax(predicted_probabilities)
    print("Predicted label index:", predicted_label_index)

    predicted_emotion = label_encoder.classes_[predicted_label_index]
    print("Predicted emotion:", predicted_emotion)


    # Emotion mapping for TESS dataset
    emotion_mapping = {
        'YAF_angry': 'ANGRY',
        'YAF_disgust': 'DISGUST',
        'YAF_fear': 'FEAR',
        'YAF_happy': 'HAPPY',
        'YAF_neutral': 'NEUTRAL',
        'YAF_pleasant_surprised': 'SURPRISED',
        'YAF_sad': 'SAD',
        'OAF_angry': 'ANGRY',
        'OAF_disgust': 'DISGUST',
        'OAF_Fear': 'FEAR',
        'OAF_happy': 'HAPPY',
        'OAF_neutral': 'NEUTRAL',
        'OAF_Pleasant_surprised': 'SURPRISED',
        'OAF_Sad': 'SAD',
    }


    recognizable_emotion = emotion_mapping.get(predicted_emotion)
    return recognizable_emotion


# In[ ]:


import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np

# Import xvfbwrapper
# from xvfbwrapper import Xvfb


# In[ ]:


class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Prediction App")
        self.root.configure(bg='yellow')

        self.emotion_to_emoji = {
            "HAPPY": "happy.png",
            "SAD": "sad.png",
            "ANGRY": "angry.png",
            "SURPRISED": "surprised.png",
            "NEUTRAL": "neutral.png",
            "FEAR": "fear.png",
            "DISGUST": "disgust.png"
        }

        self.emoji_image = None
        self.prediction_history = []

        self.show_home_page()

    def show_home_page(self):
        self.clear_window()

        label = tk.Label(self.root, text="Welcome to Emotion Prediction App", font=('Helvetica bold', 16))
        label.pack(pady=20)

        button = tk.Button(self.root, text="Audio Prediction", command=self.show_audio_page, bg='orange')
        button.pack()

        button_history = tk.Button(self.root, text="Prediction History", command=self.show_history_page, bg='lightgreen')
        button_history.pack(pady=10)

        about_button = tk.Button(self.root, text="About The App", command=self.show_about_page, bg='lightblue')
        about_button.pack(pady=10)

    def show_audio_page(self):
        self.clear_window()

        canvas = tk.Canvas(self.root, width=500, height=500, bg='skyblue')
        canvas.pack()

        label1 = tk.Label(self.root, text='SPEECH EMOTION', font=('Helvetica bold', 26))
        canvas.create_window(250, 50, window=label1)

        def upload_audio():
            file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
            if file_path:
                predicted_emotion = predict_emotion(file_path)
                label2.config(text=predicted_emotion)

                self.prediction_history.append((os.path.basename(file_path), predicted_emotion))

                emoji_image_path = self.emotion_to_emoji.get(predicted_emotion)
                if emoji_image_path:
                    emoji_image = Image.open(emoji_image_path)
                    emoji_image = emoji_image.resize((100, 100), Image.ANTIALIAS)
                    self.emoji_image = ImageTk.PhotoImage(emoji_image)
                    emoji_label.config(image=self.emoji_image)

        button1 = tk.Button(self.root, text='Upload Audio', command=upload_audio, bg='orange')
        canvas.create_window(250, 150, window=button1)

        label2 = tk.Label(self.root, text='Predicted Emotion Will Be Displayed Here')
        canvas.create_window(250, 200, window=label2)

        emoji_label = tk.Label(self.root, image=None)
        canvas.create_window(250, 300, window=emoji_label)

        back_button = tk.Button(self.root, text="Back to Home", command=self.show_home_page)
        canvas.create_window(250, 400, window=back_button)

    def show_history_page(self):
        self.clear_window()

        canvas = tk.Canvas(self.root, width=500, height=500, bg='lightgreen')
        canvas.pack()

        label = tk.Label(self.root, text="Prediction History", font=('Helvetica bold', 16))
        canvas.create_window(250, 50, window=label)

        if self.prediction_history:
            for index, (file_name, predicted_emotion) in enumerate(self.prediction_history, start=1):
                history_text = f"{index}. File: {file_name}, Emotion: {predicted_emotion}"
                history_label = tk.Label(self.root, text=history_text)
                canvas.create_window(250, 100 + index * 30, window=history_label)
        else:
            no_history_label = tk.Label(self.root, text="No prediction history available.")
            canvas.create_window(250, 150, window=no_history_label)

        back_button = tk.Button(self.root, text="Back to Home", command=self.show_home_page)
        canvas.create_window(250, 450, window=back_button)

    def show_about_page(self):
        self.clear_window()

        canvas = tk.Canvas(self.root, width=500, height=500, bg='skyblue')
        canvas.pack()

        label = tk.Label(self.root, text="About The Software", font=('Helvetica bold', 16))
        canvas.create_window(250, 50, window=label)

        about_text = ("Hello Everyone !! "
                      " Speech Emotion Recognition is a software that recognizes the emotion of the user."
                      " All of the audio files in this software should be inputted with '.wav' extension."
                      " A special thanks to the University of Toronto for the TESS data set and to all of my guiders"
                      " at clevered that guided me throughout the journey of making this software.")

        about_label = tk.Label(self.root, text=about_text, wraplength=400)
        canvas.create_window(250, 150, window=about_label)

        back_button = tk.Button(self.root, text="Back to Home", command=self.show_home_page)
        canvas.create_window(250, 400, window=back_button)

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    # Install Xvfb if not already installed
    # !apt-get update
    # !apt-get install -y xvfb  # Install using apt-get
    # !yum update  # Uncomment and use yum for CentOS/RHEL
    # !yum install -y xorg-x11-server-Xvfb  # Install using yum

    # Start Xvfb
    # vdisplay = Xvfb()
    # vdisplay.start()

    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()

    # Stop Xvfb when the application is closed
    # vdisplay.stop()


# In[ ]:





# In[ ]:




