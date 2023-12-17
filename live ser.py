import os
import time
import librosa
import pandas as pd
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import keras
from keras.models import model_from_json
from concurrent.futures import ThreadPoolExecutor

emotion_list = ['negative', 'neutral', 'positive']

# Загрузка модели
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/my_model.keras")
print("Loaded model from disk")

opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
input_duration = 3


def record_and_save(folder_path, duration=input_duration, fs=44100):
    # Записываем аудио
    recording = sd.rec(int(fs * duration), samplerate=fs, channels=2, dtype=np.int16)
    sd.wait()

    # Сохраняем отрывок аудио в папку
    file_path = os.path.join(folder_path, "record.wav")
    write(file_path, fs, recording)
    return file_path


def calculate_emotion(path, total_columns=259):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=input_duration, sr=22050 * 2,
                                  offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)

    # Создаем DataFrame с 258 столбцами и заполняем начальные значения из mfccs
    df = pd.DataFrame(columns=range(total_columns))
    df.loc[0, :len(mfccs) - 1] = mfccs

    # Заполняем оставшиеся значения 0
    df.fillna(0.0, inplace=True)
    print(df)
    print(df.columns)
    test_valid = np.array(df)
    test_valid = np.expand_dims(test_valid, axis=2)

    preds = loaded_model.predict(test_valid,
                                 batch_size=16,
                                 verbose=1)
    preds1 = preds.argmax(axis=1)
    print()
    print(preds)
    print('negative', 'neutral', 'positive')
    print(preds1)
    print()
    print(emotion_list[preds1[0]])


def main():
    folder_path = "audio"
    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            # Запуск задачи записи аудио в отдельном потоке
            future_record = executor.submit(record_and_save, folder_path)

            # Запуск задачи вычисления эмоции в отдельном потоке
            future_emotion = executor.submit(calculate_emotion, future_record.result())

            # Ожидание завершения обеих задач
            future_record.result()
            future_emotion.result()


if __name__ == "__main__":
    main()