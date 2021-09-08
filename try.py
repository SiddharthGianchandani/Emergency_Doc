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
                    c1=int(c)
                    row1.append(c1)
            
            
            train.append([np.array(row1),np.array(row[-1])])
            
    #shuffle(train)
    print(train[:1])
    np.save('train_data.npy',train)
    return train

train()
