from sklearn.preprocessing import OneHotEncoder, LabelEncoder

items = []
with open('LOC_synset_mapping.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()

        line = line.split()
        items.append(line[0])

# print(items)

#Step1: 모든 문자를 숫자형으로 변환합니다.
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)

#Step2: 2차원 데이터로 변환합니다.
labels = labels.reshape(-1, 1)

#Step3: One-Hot Encoding 적용합니다.
oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
vector = oh_labels.toarray()
# print(oh_labels)
# print(oh_labels.toarray())
# print(oh_labels.shape)