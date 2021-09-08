import os
import csv
import json
from random import shuffle
import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

LR=1e-3
MODEL_NAME='NotDelicious-{}-{}.model'.format(LR,'2conv-basic-video')

def train():
    train=[]
    with open("dataset.csv") as s:
        thereader=csv.reader(s)
        for row in thereader:
            age=row[0]

            if age=="child":
                age='0000'
            elif age=="adult1":
                age='0001'
            elif age=="adult2":
                age='0010'
            elif age=="adult3":
                age='0011'
            elif age=="adult4":
                age='0100'
            elif age=="senior citizen":
                age='0101'
            elif age=="toddler":
                age='0110'

            row[0]=age
            gender=row[1]

            if gender=="female":
                gender="0001"
            elif gender=="male":
                gender="0010"

            row[1]=gender
            intensity=row[2]

            if intensity=="mild":
                intensity="0000"
            elif intensity=="moderate":
                intensity="0001"
            elif intensity=="extreme":
                intensity="0010"

            row[2]=intensity
            ail=row[3]

            if ail=="fever":
                ail="0000"
            elif ail=="cold":
                ail="0001"
            elif ail=="headache":
                ail="0010"
            elif ail=="infection":
                ail="0011"
            elif ail=="joint pain":
                ail="0100"
            elif ail=="irritable bowel syndrome":
                ail="0101"
            elif ail=="period pain":
                ail="0110"
            elif ail=="ADHD":
                ail="0111"
            elif ail=="anxiety disorder":
                ail="1000"
            elif ail=="panic disorder":
                ail="1001"
            elif ail=="urethritis" or ail=="cervicitis":
                ail="1010"
            elif ail=="acne":
                ail="1011"
            elif ail=="syphilis":
                ail="1100"
            elif ail=="gonorrhea":
                ail="1101"
            elif ail=="chlamydia":
                ail="1110"
            elif ail=="binge eating disorder":
                ail="1111"
                
            row[3]=ail
            cure=row[4]

            if cure=="paracetamol":
                cure=[0,0,0,0,0,0,0,1]
            elif cure=="amoxicillin":
                cure=[0,0,0,0,0,0,1,0]
            elif cure=="cyclopam":
                cure=[0,0,0,0,0,1,0,0]
            elif cure=="adderall":
                cure=[0,0,0,0,1,0,0,0]
            elif cure=="xanax":
                cure=[0,0,0,1,0,0,0,0]
            elif cure=="azithromycin":
                cure=[0,0,1,0,0,0,0,0]
            elif cure=="tetracycline":
                cure=[0,1,0,0,0,0,0,0]
            elif cure=="vyvanse":
                cure=[1,0,0,0,0,0,0,0]
             

            row[4]=cure
            
            row1=[]
            for r in row[:-1]:
                for c in r:
                    row1.append(c)
            
            train.append([np.array(row1),np.array(row[-1])])
            
    shuffle(train)
    np.save('train_data.npy',train)
    return train

def read_json(name):
    with open('show.JSON') as s:
        data=json.load(s)
        temp=data[name]
        for d in temp:
            print("    ",end='')
            print(d,end=': ')
            print(temp[d])

convnet = input_data(shape=[None, 4, 4,1], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.7)

convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)

#training=train()
training=np.load('train_data.npy',allow_pickle=True)

testing=training[-20:]
training=training[:-20]
'''
train=training[:-50]
test=training[-50:]
 
X=np.array([i[0] for i in train]).reshape(-1,4,4,1)
Y=[i[1] for i in train]
 
test_x=np.array([i[0] for i in test]).reshape(-1,4,4,1)
test_y=[i[1] for i in test]

print("X")
print(X[:20])
print("Y")
print(Y[:20])
print("test_x")
print(test_x[:20])
print("test_y")
print(test_y[:20])


model.fit({'input': X}, {'targets': Y}, n_epoch=800, validation_set=({'input': test_x}, {'targets': test_y}), 
          snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)
'''
test=np.array([i[0] for i in testing]).reshape(-1,4,4,1)
for data in test:
    temp=np.array([i for i in data]).reshape(-1,4,4,1)

    model_out=model.predict(temp)[0]

    age_temp=temp[0][0]
    age=''
    for a in age_temp:
        for c in a:
            age+=c

    if age=="0000":
        age='child'
    elif age=="0001":
        age='adult1'
    elif age=="0010":
        age='adult2' 
    elif age=="0011":
        age='adult3'
    elif age=="0100":
        age='adult4'
    elif age=="0101":
        age='senior citizen'
    elif age=="0110":
        age='toddler'
    print(age)

    gender_temp=temp[0][1]
    gender=''
    for g in gender_temp:
        for c in g:
            gender+=c

    if gender=="0001":
        gender="female"
    elif gender=="0010":
        gender="male"
    print(gender)
    
    intensity_temp=temp[0][2]
    intensity=''
    for i in intensity_temp:
        for c in i:
            intensity+=c

    if intensity=="0000":
        intensity="mild"
    elif intensity=="0001":
        intensity="moderate"
    elif intensity=="0010":
        intensity="extreme"
    print(intensity)

    ail=''
    ail_temp=temp[0][3]    
    for a in ail_temp:
        for c in a:
            ail+=c

    if ail=="0000":
        ail="fever"
    elif ail=="0001":
        ail="cold"
    elif ail=="0010":
        ail="headache"
    elif ail=="0011":
        ail="infection"
    elif ail=="0100":
        ail="joint pain"
    elif ail=="0101":
        ail="irritable bowel syndrome"
    elif ail=="0110":
        ail="period pain"
    elif ail=="0111":
        ail="ADHD"
    elif ail=="1000":
        ail="anxiety disorder"
    elif ail=="1001":
        ail="panic disorder"
    elif ail=="1010":
        if gender=="male":
            ail="urethritis"
        else:
            ail="cervicitis"
    elif ail=="1011":
        ail="acne"
    elif ail=="1100":
        ail="syphilis"
    elif ail=="1101":
        ail="gonorrhea"
    elif ail=="1110":
        ail="chlamydia"
    elif ail=="1111":
        ail="binge eating disorder"
    print(ail)
    
    
    print(model_out)
    temp=[]
    for m in model_out:
        if m>=0.1:
            temp.append(m)

    if len(temp)>1:
        temp=temp[:2]
    print(temp)

    ans=[]
    for m in model_out:
        if m in temp:
            ans.append(1)
        else:
            ans.append(0)

    answer=[]
    if ans[0]==1:
        answer.append("vyvanse")
    if ans[1]==1:
        answer.append("tetracycline")
    if ans[2]==1:
        answer.append("azithromycin")
    if ans[3]==1:
        answer.append("xanax")
    if ans[4]==1:
        answer.append("adderall")
    if ans[5]==1:
        answer.append("cyclopam")
    if ans[6]==1:
        answer.append("amoxicillin")
    if ans[7]==1:
        answer.append("paracetamol")

    print("The recommendation is: ")    
    for a in answer:
        print(a)
        read_json(a)
        print('---------------------')
    c=input("Next?\n--->")
    if c=="y":
        continue
    else:
        break

print("See you next time")
