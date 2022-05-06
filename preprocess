import numpy as np
import pandas as pd
import xlwt
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

txtpath='D:\\Data\\1W\\circRNA_dl_datanew.txt'   ####Text file to be processed
savepath='D:\\Data\\1W\\circRNAnew.xlsx'   ####Save Path after processing
#----------------------Sequence Data pre-processing----------
def txt_xlsx():
   try:
     file = open(txtpath)
     xls=xlwt.Workbook()
     sheet=xls.add_sheet('sheet1',cell_overwrite_ok=True)
     x=0
     while True:
         line=file.readline()
         if not line:
            break
         for i in range(len(line.split('\t'))):
            item=line.split('\t')[i]
            sheet.write(x,i,item)
         x+=1
     file.close()
     xls.save(savepath)
   except:
     raise

txt_xlsx()
file=pd.read_excel(savepath,skiprows=np.arange(3,34896,3),names=['species','label','seq'])
file=pd.DataFrame(file) ##Transfer to the dataframe
print(file)
file1=file[:23266:2]  ##Take the corresponding line according to step length
file2=file[1:23265:2]
print(file1['seq'])
index=np.arange(1,11633,1)##Reset the index
file1.set_index(index,inplace=True)  ##Overwrite the original index
file2.set_index(index,inplace=True)
file1.insert(3,'labelMo',file2['label'])
file1.insert(4,'seq2',file2['seq'])
# #----------------Processing Annotation data--------
annotationtxtpath='D:\\Data\\1W\\circRNA_dl_annotationnew.txt'
annotatiponsavepath='D:\\Data\\1W\\circRNAannotationnew.xlsx'
def txt_xlsx2():
   try:
     file = open(annotationtxtpath)
     xls=xlwt.Workbook()
     sheet=xls.add_sheet('sheet1',cell_overwrite_ok=True)
     x=0
     while True:
         line=file.readline()
         if not line:
            break
         for i in range(len(line.split('\t'))):
            item=line.split('\t')[i]
            sheet.write(x,i,item)
         x+=1
     file.close()
     xls.save(annotatiponsavepath)
   except:
     raise
txt_xlsx2() ###Run function
fileann=pd.read_excel(annotatiponsavepath,skiprows=[1],names=['sequence ID','gene ID','circRNA position'])
fileann=pd.DataFrame(fileann) ##Transfer to the dataframe

#-----------------Run the genetic numbers on both files, process the file labels--------
for a in np.arange(1,fileann.shape[0],1):
   x=0
   for i in fileann['sequence ID']:
       if file1.at[a,'label']==i:
           x=1
       else:
           x=x
   if x==1:
       file1.at[a,'label']=1
   else:
       file1.at[a,'label']=0
for a in np.arange(1,11633,1):
   x=0
   for i in fileann['sequence ID']:
       if file1.at[a,'labelMo']==i:
           x=1
       else:
           x=x
   if x==1:
       file1.at[a,'labelMo']=1
   else:
       file1.at[a,'labelMo']=0
data=file1
labelsavepath='D:\\Data\\2W\\BZlabel1003.npy'
data1=np.array(data['label']).reshape(11632,1)
np.save(labelsavepath,data1)
# #---------------- Sequence data segmentation--------

data = np.array(pd.read_excel(savepath))
trans = []
for i in data:
    trans.append(list(i[0]))
trans = np.array(trans).transpose()
trans=np.array(trans)
print(trans.shape)
np.save('D:\\Data\\seq.npy',trans)
dataseq1=np.delete(data,15000,axis=0)### Delete ‘\n’
dataseq1=dataseq1[5000:15000]###A sequence of 1W bp
ai1=np.array([])
for i in range(dataseq.shape[0]): ####Find the sequence where ‘N’exists
    for a in dataseq1[:,i]:
        if a=='N':
            ai1=np.append(ai1,i).astype(np.int32)
            break
label=np.load('D:\\Data\\crRNA\\label.npy')
data1=np.load('D:\\Data\\seq.npy')
data1=np.delete(data1,ai1,axis=1)###Deleate  Sequences Contains 'N'
label1=np.delete(label,ai1,axis=1)
label1=tf.one_hot(labe1,depth=2)
alphabet = 'ATCG-'
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
for i in range(10000):
   dataall[i,:] = [char_to_int[char] for char in dataall[i,:]] ##Character to integer encoding
data1=np.array(data1,dtype='int')
data1=data1.T
data=tf.one_hot(data1,depth=5)
data=data.transpose(0,2,1)

np.save('D:\\Data\\2W\\newlabel.npy',label1)  ###Storage label
np.save('D:\\Data\\2W\\allxnew.npy',data)###Store the new sequence data file
