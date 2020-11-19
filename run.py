# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:53:23 2020

@author: BRASLab
"""

import shutil
import os
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import sounddevice as sd
import librosa
import wave
import pyaudio
from scipy.io.wavfile import write
import recording as record
from recording_vad import vad
from scipy.io.wavfile import read
from CQCC import cqt_cqcc
from CQCC import CNN_cm_predict
from PIL import Image
#from CQCC import DNN_cm_predict
import test_iv
import main_multi_predict as mul_predict
import main_iv_one_predict as one_predict
import main_iv_cnn_predict as cnn_predict 
import svm_iv_classifier as svm
import randomforest_classifier as randomforest
import plda_scoring 
import streamlit as st


lock = open("lock.txt", "r")
lock_number = lock.read()
lock_number = int(lock_number)
lock.close()
if lock_number == 3:
    st.warning("您已經錯誤3次，請解鎖")
    if st.button("解鎖"):
        lock_number = 0
        lock = open("lock.txt", "w")
        lock.write(str(lock_number))
        lock.close()
    st.stop()
if lock_number == 4:
    if st.button("解鎖"):
        lock_number = 0
        lock = open("lock.txt", "w")
        lock.write(str(lock_number))
        lock.close()
    st.warning("您是欺騙攻擊，請解鎖")
    st.stop()
if lock_number == 5:
    if st.button("解鎖"):
        lock_number = 0
        lock = open("lock.txt", "w")
        lock.write(str(lock_number))
        lock.close()
    st.warning("您是冒充者，請解鎖")
    st.stop()



multi_modelList = 'multimodel_h5'
one_modelList = 'onemodel_h5'
cnn_modelList = 'cnnmodel_h5'
cqcc_modelList = 'CQCC/model'
test_feat_dir = 'data/test_feat'
test = 'data/test'
name = 0 

def model_class(total_class,total_speaker_count):
    
    for i in range(total_speaker_count.shape[0]):
        for j in range(total_speaker_count.shape[1]):
            total_speaker_count[i][j]=list(total_class[i]).count(j)
    
    return total_speaker_count
@st.cache(allow_output_mutation=True)
def load():
    Asv_model = load_model('./{0}/Asv.h5'.format(multi_modelList))
    Asv_model._make_predict_function()
    kao_model = load_model('./{0}/kao.h5'.format(multi_modelList))
    kao_model._make_predict_function()
    Lu_model = load_model('./{0}/Lu.h5'.format(multi_modelList)) 
    Lu_model._make_predict_function()  

    one_model = load_model('./{0}/enroll_one.h5'.format(one_modelList)) 
    one_model._make_predict_function()

    cnn_model = load_model('./{0}/enroll_cnn.h5'.format(cnn_modelList))
    cnn_model._make_predict_function()
    cnn_cqcc_model = load_model('./{0}/CQCC_cnnmodel.h5'.format(cqcc_modelList))
    cnn_cqcc_model._make_predict_function()
    spoof_sc = joblib.load('CQCC/std_scaler.bin')    
    trainLabel = np.load('data/train_model/enroll_label.npy')
    speaker_count = np.unique(trainLabel)
    return Asv_model, kao_model, Lu_model, one_model, cnn_model, cnn_cqcc_model, spoof_sc, speaker_count

if __name__ == "__main__":

    title = "Speaker Recognition"
    st.title(title)
    image = Image.open('speak-the-words.png')
    st.image(image, use_column_width=True)

    if st.checkbox('模型載入'):
        if lock_number == 3:
            st.stop()
        Asv_model, kao_model, Lu_model, one_model, cnn_model, cnn_cqcc_model, spoof_sc, speaker_count = load()


    if st.button('錄音測試'):        
        with st.spinner(f'Recording for 3 seconds ....'):
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            RECORD_SECONDS = 3
            WAVE_OUTPUT_FILENAME = "."

            p = pyaudio.PyAudio()

            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            print("* recording")

            frames = []

            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("* done recording")

            stream.stop_stream()
            stream.close()
            p.terminate()

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

        st.success("Recording completed")
        audio_file = open(WAVE_OUTPUT_FILENAME, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)

    if st.checkbox('辨識'):
        ##process CQCC
        spoof_class = []
        count = 0
        test_path = os.listdir(test)
        for wavfile in test_path:
            fs, sig = read(test + '/' + wavfile)
            sig = sig / 32768
            sig = (sig - np.mean(sig))
            
            test_cqcc = cqt_cqcc.CQCC(sig,fs)
            test_cqcc = cqt_cqcc.frame_split(test_cqcc)
            cm_cnn_class = CNN_cm_predict.cm_cnn_predict(cnn_cqcc_model,test_cqcc,spoof_sc)
#            cm_dnn_class = DNN_cm_predict.cm_one_predict(dnn_cqcc_model,test_cqcc)
            spoof_class.append(cm_cnn_class)
        print(spoof_class)
        for i in range(0,len(cm_cnn_class),1):
            if cm_cnn_class[i] in [0]: #偵測聲音中的攻擊 0:攻擊語音 1:真實語音
                count+=1
        if count >0 :
            print('spoof')
#                os.remove('data/test/test_{}.wav'.format(name))
#                name+=1
            st.write("spoof")
            lock = open("lock.txt", "w")
            lock_number = 4
            lock.write(str(lock_number))
            lock.close()
        elif count <=0 :
            vad_threshold = vad()
#            if vad_threshold != True : # vad沒大於threshold 跳出
#                continue
        
#               import timeit
#               start = timeit.default_timer()
            test_iv.create_idMap("test")
            test_iv.extract_features("test")
            test_iv.stat_servers()
            test_iv.i_vector("test")
#               stop = timeit.default_timer()
#               print('extractor i-vector time')
#               print(stop - start)
            os.remove('data/test_idmap.h5')
#               os.remove('data/test/test_{}.wav'.format(name))
#               name+=1
            shutil.rmtree(test_feat_dir)
            
            total_class=[]
            mul_predict_class = mul_predict.multi_predict(Asv_model,kao_model,Lu_model)
            one_predict_class = one_predict.one_predict(one_model)
            cnn_predict_class = cnn_predict.cnn_predict(cnn_model)
            svm_linear_predict_class = svm.svm_linear_predict()
            svm_rbf_predict_class = svm.svm_rbf_predict()
            randomforest_predict_class = randomforest.randomforest_predict()
#               print('multi predict')            
#               print(mul_predict_class)
#               print('one predict')
#               print(one_predict_class)
#               print('cnn predict')
#               print(cnn_predict_class)
#               print('svm predict')
#               print(svm_predict_class)

            total_class = np.vstack((mul_predict_class,one_predict_class,cnn_predict_class,svm_linear_predict_class,svm_rbf_predict_class,randomforest_predict_class))#           
#            total_class = np.vstack((one_predict_class,cnn_predict_class,svm_linear_predict_class,svm_rbf_predict_class,randomforest_predict_class))#           

            total_speaker_count = np.zeros((total_class.shape[1],np.size(speaker_count)+1)) #+1 為了分類出冒充者
            total_class = total_class.T   #將class轉成count
            total_speaker_count = model_class(total_class,total_speaker_count) #statistics speaker classify
            print(total_class)
            print(total_speaker_count.shape)  

            for i in range(total_speaker_count.shape[0]):
                if total_speaker_count[i][0]==1: #[0] 0 
                    print('retry')
                    st.write("retry")                    
                    lock_number +=1
                    st.warning(f"您已經錯誤{lock_number}次")
                    lock = open("lock.txt", "w")
                    lock.write(str(lock_number))
                    lock.close()
                elif total_speaker_count[i][0]>1: 
                    print('imposter')
                    st.write("imposter")
                    lock = open("lock.txt", "w")
                    lock_number = 5
                    lock.write(str(lock_number))
                    lock.close()
                elif np.max(total_speaker_count[i,1:])==6:  #5個model投票同一個語者
                    print(np.argmax(total_speaker_count[i,1:])+1)
                    st.write(np.argmax(total_speaker_count[i,1:])+1)
                else:
                    plda_speaker_diff = np.where(total_speaker_count[i,1:]>=1,1,0)
                    
                    plda_scoring.create_test_trials()
                    plda_scoring.create_Ndx()
                    speaker,score = plda_scoring.plda_score(i,plda_speaker_diff)
                    if score <= 0.0:
                        print('retry')
                        st.write("retry")
                        lock_number +=1
                        st.warning(f"您已經錯誤{lock_number}次")
                        lock = open("lock.txt", "w")
                        lock.write(str(lock_number))
                        lock.close()
                    else:
                        print(speaker)
                        st.write(speaker)
                        print(score)
                        st.write(f"score:{score}")
            
#                       plda_speaker = np.argmax(total_speaker_count[i,1:])+1
#                    
#                       print(np.argmax(total_speaker_count[i,1:])+1) #求出最大值的索引
#                       print(score)
                    
                    
                    
                    
            
            
            