import os
import csv

def modify(rows):
    with open("dataset.csv",'w',newline='') as s:
        thewriter=csv.writer(s)
        for row in rows:
            thewriter.writerow(row)
        
def train():
    rows=[]
    with open("dataset.csv") as s:
        thereader=csv.reader(s)
        for row in thereader:
            temp=int(row[0])
            if temp<12:
                temp="toddler"
            elif 12<=temp<=17:
                temp="child"
            elif 30>=temp>=18:
                temp="adult1"
            elif 40>=temp>30:
                temp="adult2"
            elif 50>=temp>40:
                temp="adult3"
            elif 60>temp>50:
                temp="adult4"
            elif temp>=60:
                temp="senior citizen"
            row[0]=temp
            print(row)
            rows.append(row)
        modify(rows)
        return
train()

