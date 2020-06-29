#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
import model.ml_model as prediction



features = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login',
'is_guest_login',
'count',
'srv_count',
'serror_rate',
'srv_serror_rate',
'rerror_rate',
'srv_rerror_rate',
'same_srv_rate',
'diff_srv_rate',
'srv_diff_host_rate',
'dst_host_count',
'dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'intrusion_type']

print("Feature size: ", len(features))


data = pd.read_csv('dataset/kddcup/kddcup.data_10_percent', names=features, header=None)
data.head()


print('The no of data points are:',data.shape[0])
print('The no of features are:',data.shape[1])
print('Some of the features are:',features[:10])


output = data['intrusion_type'].values
labels = set(output)


print('The different type of output labels are:',labels)


# Data Cleaning
print('Null values in the dataset are: ', len(data[data.isnull().any(1)]))
data.drop_duplicates(subset=features, keep='first', inplace=True)
data.shape


plt.figure(figsize=(20,15))
class_distribution = data['intrusion_type'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()


plt.figure(figsize=(20,15))
sns.violinplot(x="intrusion_type", y="src_bytes", data=data)
plt.xticks(rotation=90)

# feature extraction
data['num_outbound_cmds'].value_counts()
data.drop('num_outbound_cmds', axis=1, inplace=True)
data['is_host_login'].value_counts()
data.drop('is_host_login', axis=1, inplace=True)


# Transformation of categorical values
data['protocol_type'] = data['protocol_type'].astype('category')
data['service'] = data['service'].astype('category')
data['flag'] = data['flag'].astype('category')
cat_columns = data.select_dtypes(['category']).columns
data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)


X = data.drop('intrusion_type', axis=1)
Y = data['intrusion_type']

data.replace(to_replace = ['ipsweep.', 'portsweep.', 'nmap.', 'satan.'], value = 'probe', inplace = True)
data.replace(to_replace = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.'], value = 'r2l', inplace = True)
data.replace(to_replace = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.'], value = 'u2r', inplace = True)
data.replace(to_replace = ['back.', 'land.' , 'neptune.', 'pod.', 'smurf.', 'teardrop.'], value = 'dos', inplace = True)


# Standardization
sScaler = StandardScaler()
rescaleX = sScaler.fit_transform(X)
print(rescaleX)
names_inputed =features[0:39]
data = pd.DataFrame(data=rescaleX, columns=names_inputed)
print(len(names_inputed))

# Normalization
norm = Normalizer()
X = norm.fit_transform(X)


X = np.array(X)
Y = np.array(Y)

if __name__ == "__main__":
    ml_model = prediction.Model(X, Y)
    ml_model.decision_tree()
    ml_model.random_forest()
    ml_model.naive_bayes()
    ml_model.k_nearest_neighbor()
