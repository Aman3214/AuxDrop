import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

original_csv_path = 'adult_income_dataset/adult.csv'
clean_csv_path = 'adult_income_dataset/adult_clean.csv'
df = pd.read_csv(original_csv_path)

# filling missing values
df.replace('?', pd.NA, inplace=True)

# normalizing values
scaler = MinMaxScaler()
df['age'] = scaler.fit_transform(df[['age']])
df['fnlwgt'] = scaler.fit_transform(df[['fnlwgt']])
# Encoding categorical features
categorical_features = ["workclass","education","marital-status","occupation","relationship","race","gender","native-country","income"]
label_encoders = {}
for feature in categorical_features:
    label_encoder = LabelEncoder()
    non_missing = df[feature].dropna()
    df.loc[df[feature].notna(), feature] = label_encoder.fit_transform(non_missing)
    
    # Fill NaNs with -1
    df[feature] = df[feature].fillna(-1).astype(int)
    label_encoders[feature] = label_encoder
df.to_csv(clean_csv_path,float_format="%.3f",index=False)

## uncomment for analyzing which are the auxiliary features and base feature 
# for feature in features:
#     flag=False
#     for value in  df[feature]:
#         if value==-1:
#             flag=True
#             print(feature,flag)
#             break
features = {"age","workclass","fnlwgt","education","educational-num","marital-status","occupation","relationship","race","gender","capital-gain","capital-loss","hours-per-week","native-country"}
aux_features = {"native-country","workclass","occupation"}
base_features = features - aux_features

# split data
train, test = train_test_split(df, test_size=0.2, random_state=58)
train.to_csv('adult_income_dataset/adult_train.csv',float_format="%.3f",index=False)
test.to_csv('adult_income_dataset/adult_test.csv',float_format="%.3f",index=False)

# preparing dataloader
class AdultDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data[list(features)].values
        self.x_base = self.data[list(base_features)].values  # Base features
        self.x_aux = self.data[list(aux_features)].values  # Auxiliary features
        self.y = self.data['income'].values  # Assuming 'income' is the target


        # Convert to torch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.x_base = torch.tensor(self.x_base, dtype=torch.float32)
        self.x_aux = torch.tensor(self.x_aux, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)  # Assuming classification (categorical labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the feature vector and label as a tuple
        return self.features[idx], self.x_base[idx], self.x_aux[idx], self.y[idx]


