import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns


df = pd.read_csv('./Data/cleveland.csv', header=None)

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'target']

df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())

# Nhóm dữ liệu theo độ tuổi và khả năng bị bệnh tim
age_counts = df.groupby(['age','target']).size().reset_index(name = 'count')


# ve bieu do cot

plt.figure(figsize=(12, 6))
sns.barplot(data = age_counts, x = 'age', y = 'count', hue ='target', palette= {0: 'blue', 1 : 'red'}) 
plt.title("Số lượng người bị mắc bệnh tim")

plt.xlabel("Do tuoi")
plt.ylabel("So luong nguoi")
plt.xticks(rotation = 45)
legend= plt.legend(title='Khả năng bị bệnh tim', labels=['Không' , 'Có'])
legend.get_texts()[0].set_color('blue')
legend.get_texts()[1].set_color('red')
plt.grid(axis='y')
plt.show()

#dùng hệ thống và thư viện ây quyết định để dự đoán
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d.proj3d import rotation_about_vector

#tách dữ liệu vào x và vào nhãn
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values   # Cột 'target'

#chia tep du lieu

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42)

#khoi tao mo hinh cay quyet dinh

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=2)

#huan luyen mo hinh
clf.fit(X_train, y_train)

#du doan tren tap huan luyejn va kiem thu
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Tính ma trận nhầm lẫn cho tập huấn luyện và kiểm thử
# cm_train = confusion_matrix(y_train, y_train_pred)
# cm_test = confusion_matrix(y_test, y_test_pred)
#
# # Tính toán độ chính xác cho tập huấn luyện và kiểm thử
# accuracy_for_train = np.round((cm_train[0][0] + cm_train[1][1]) / len(y_train), 2)
# accuracy_for_test = np.round((cm_test[0][0] + cm_test[1][1]) / len(y_test), 2)
#
# # In kết quả
# print('Accuracy for training set for Decision Tree = {}'.format(accuracy_for_train))
# print('Accuracy for test set for Decision Tree = {}'.format(accuracy_for_test))
# du doan cho 1 benh nhan moi
new_patient = np.array([[63, 1, 111, 140, 220, 0, 1, 160, 0, 2.3, 1, 0, 1]])  #
prediction = clf.predict(new_patient)


if prediction[0] == 1:
    print("Bệnh nhân có khả năng mắc bệnh tim.")
else:
    print("Bệnh nhân không có khả năng mắc bệnh tim.")

# su dung thuat toan thu cong

#y  một danh sách hoặc mảng các nhãn (label).
def entropy(y) :
    value_counts = np.unique(y, return_counts= True) [1] # tra ve so lan xuat hien
#Các giá trị độc nhất trong y.
#Số lần xuất hiện của mỗi giá trị (là return_counts=True).
    probabilities = value_counts / len(y)
    #chia số lần xuất hiện của từng nhãn cho tổng số lượng các phần tử trong y.
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))
def information_gain(X, y, feature_index):
    total_entropy = entropy(y) # mức độ hỗn loạn của dữ liệu (là dữ liệu chưa tinh khiết)
    values, counts = np.unique(X[:, feature_index], return_counts= True) # lấy tấy cả các giá trị khác nhau của đặc trưng tại feature_index và đếm số lượng
    weihted_entropy = 0
    for v in values:
        subset_y = y[X[:, feature_index] == v]
        if len(subset_y) > 0:
            weihted_entropy += (len(subset_y)/len(y)) * entropy(subset_y)
    return total_entropy - weihted_entropy

def id3(X, y, features):
    # Neu tat ca deu giong nhau, tra ve nhan do
    if len(np.unique(y)) == 1 :
        return np.unique(y) [0] # tra ve gia tri duy nhat dau tien
# được sử dụng để lấy giá trị duy nhất đầu tiên từ mảng y.
    # Nếu mảng y chỉ chứa một giá trị duy nhất, thì nó sẽ trả về giá trị đó.


    #neu k con thuoc tinh de chia, tra ve nhan pho bien nhat
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    #tinh gain cho tat ca thuoc tinh
    #neu gains cao nhat thi duoc lam nut chia tiep theo trong cay quyet dinh
    gains = [information_gain(X, y, feature) for feature in features]
    #features là các thuộc tính
    #sau khi tính toán, sẽ cho kết quả từng thuộc tính 1
    best_feature_index = features[np.argmax(gains)]

    #tao cay quyet dinh


    tree = {best_feature_index: {}}
    #tạo ra từ điển, lưu những giá trị thuộc thuộc tính gain to nhất (là 1 từ điển)
    for value in np.unique(X[:, best_feature_index]):
        #duyệt qua từng giá trị duy nhất của thuộc tính tốt nhất
         subset_indices = X[:, best_feature_index] == value
#Tạo ra 1 mảng boolen, xác định hàng nào trong x có giá trịc thuộc tính tốt nhấ = với giá trị đang lắp qua
        #ví dụ sau khi tinh toán, ta chọn được huyết áp
         subset_X = X[subset_indices]
        #lay ra cac thuộc  tính được chọn trong huyết áp như [45, 140, cao , có]
         subset_y = y[subset_indices]
        # lay ra cac nhãn duoc chon tren subset_indices
         #Tao cay con bang cach goi de quy
         subTree = id3(subset_X ,subset_y,[f for f in features if f != best_feature_index] )

         tree[best_feature_index][value] = subTree
#tree = { 'Huyết áp': { 140: 'Có' } }
    # cuối cùng, sẽ cho ra dạng
   # tree = {
    #    'Huyết áp': {
     #       120: 'Có',
      #      130: 'Không',
       #     135: 'Không',
        #    140: 'Có',
         #   150: 'Có'
        #}
   # }
    return tree

features = list(range(X.shape[1]))
decision_tree = id3(X, y , features)

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    feature_index = list(tree.keys())[0]
    feature_value = sample[0][feature_index]

    print(f'Kiem tra thuoc tinh {feature_index} voi gia tri {feature_value}')

    for value in tree[feature_index]:
        if feature_value == value:
            return predict(tree[feature_index][value], sample)
    return None


manual_prediction = predict(decision_tree, new_patient)
if manual_prediction is not None:
    print("Dự đoán từ cây quyết định thủ công, bệnh nhân ", "Có khả năng mắc bệnh tim" if manual_prediction == 1 else "Không có khả năng mắc bệnh tim")
else:
    print("Dự đoán từ cây quyết định thủ công không thành công.")