import numpy as np
import pandas as pd
import random
random.seed(42)
np.random.seed(42)
from glob import glob
import os
import sys
import xml.etree.ElementTree

if not os.path.exists('csvs'):
        os.makedirs('csvs')

tags = ['ButtonCircle', 'ButtonSquare', 'Text', 'TextInput', 'ImageView', 'RadioButton', 'CheckBox']

xml_files_path = './test/test_xml/'

xml_files = [f for f in os.listdir(xml_files_path) if os.path.isfile(os.path.join(xml_files_path, f))]
xml_labels = np.full([len(xml_files), 7], -1)
#xml_files.sort()

for f in xml_files:
    e = xml.etree.ElementTree.parse(xml_files_path + f).getroot()
    for object in e.findall('object'):
        for tag in tags:
            name = object[0].text
            if name == tag:
                xml_labels[xml_files.index(f)][tags.index(tag)] = 1

xml_files = np.array([[os.path.splitext(f)[0] for f in xml_files]])

xml = np.concatenate((xml_files.T, xml_labels), axis=1)

#print(xml_labels[0])
#print(xml_files[0])
#print(xml[0])

df = pd.DataFrame(data=xml, columns=['Files'] + tags, index=None)
#df = df.drop(df.columns[0], axis=1)
df.to_csv('csvs/test_all.csv', index=False)
for tag in tags:
    new = df[['Files', tag]]
    ad = int(0.4 * new.shape[0])
    ag = int(0.8 * new.shape[0])
    new_train = new.iloc[:ad]
    new_trainval = new.iloc[:ag]
    new_valid = new.iloc[ad:ag]
    new_test = new.iloc[ag:]
    new_train.to_csv('csvs/{}_train.txt'.format(tag), header=None, index=False, columns=None, sep=' ', mode='a')
    new_valid.to_csv('csvs/{}_trainval.txt'.format(tag), header=None, index=False, columns=None, sep=' ', mode='a')
    new_valid.to_csv('csvs/{}_val.txt'.format(tag), header=None, index=False, columns=None, sep=' ', mode='a')
    new_test.to_csv('csvs/{}_test.txt'.format(tag), header=None, index=False, columns=None, sep=' ', mode='a')