import gc
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

for i in tqdm(range(4)):
    test_image_data = pd.read_parquet('data/test_image_data_{}.parquet'.format(i), engine='pyarrow')
    test_df = pd.merge(test_image_data, test, on='image_id').drop(['image_id'], axis=1)

del test
del test_image_data

def resize(df):
    resized = {}
    resize_size=64
    for i in tqdm(range(df.shape[0])):
        img = df.iloc[i, :].values.reshape(137, 236)
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
        resized[i] = img.reshape(-1)
    resized_df = pd.DataFrame(resized).T
    return resized_df

SIZE=64
CHANNELS=1
x_test = resize(test_df.drop(['row_id', 'component'], axis=1))/255.0
x_test = x_test.values.reshape(-1, SIZE, SIZE, CHANNELS)

model = tf.keras.models.load_model('bengali-cv19-model.h5')

model.summary()

lr_reducer_grapheme = tf.keras.callbacks.ReduceLROnPlateau(monitor='output_grapheme_accuracy', 
                                    factor=np.sqrt(0.1),
                                    patience=5, 
                                    verbose=1,
                                    min_lr=0.5e-6)
lr_reducer_vowel = tf.keras.callbacks.ReduceLROnPlateau(monitor='output_vowel_accuracy', 
                                     factor=np.sqrt(0.1),
                                     patience=5, 
                                     verbose=1,
                                     min_lr=0.5e-6)
lr_reducer_consonant = tf.keras.callbacks.ReduceLROnPlateau(monitor='output_consonant_accuracy', 
                                         factor=np.sqrt(0.1),
                                         patience=5, 
                                         verbose=1, 
                                         min_lr=0.5e-6)

callbacks = [lr_reducer_grapheme, lr_reducer_vowel, lr_reducer_consonant]

del lr_reducer_grapheme
del lr_reducer_consonant
del lr_reducer_vowel

batch_size=32
prediction = model.predict(x_test, batch_size=batch_size, verbose=1, callbacks=callbacks)

prediction_dict = {
    'grapheme_root': [],
    'vowel_diacritic': [],
    'consonant_diacritic': []
}

components = ['consonant_diacritic', 'grapheme_root', 'vowel_diacritic']
target=[]
row_id=[]

for i, p in enumerate(prediction_dict):
    prediction_dict[p] = np.argmax(prediction[i], axis=1)

for k,id in enumerate(test_df.index.values):
    for i,comp in enumerate(components):
        id_sample='Test_'+str(id)+'_'+comp
        row_id.append(id_sample)
        target.append(prediction_dict[comp][k])

del test_df
del prediction
del prediction_dict
gc.collect()

df = pd.DataFrame({'row_id':row_id, 'target':target}, columns=['row_id','target'])
df.to_csv('submission.csv', index=False)
