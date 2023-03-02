from text_preprocess import text_preprocess
from remove_stopwords import remove_stopwords
from sklearn.preprocessing import LabelEncoder
import pickle
import os

MODEL_PATH = "models"

# Chia tập train/test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
test_percent = 0.2

text = []
label = []

for line in open('train_model/news_categories.prep',encoding='utf-8'):
    words = line.strip().split()
    label.append(words[0])
    text.append(' '.join(words[1:]))

X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=test_percent)

# Lưu train/test data
# Giữ nguyên train/test để về sau so sánh các mô hình cho công bằng
with open('train.txt', 'w',encoding='utf-8') as fp:
    for x, y in zip(X_train, y_train): #join hai đoạn dữ liệu vào một file
        fp.write('{} {}\n'.format(y, x))

with open('test.txt', 'w',encoding='utf-8') as fp:
    for x, y in zip(X_test, y_test): #join hai đoạn dữ liệu vào một file
        fp.write('{} {}\n'.format(y, x))

# encode label
label_encoder = LabelEncoder() #chuyển label từ text sang số
label_encoder.fit(y_train)
print(list(label_encoder.classes_), '\n')
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# print(X_train[0], y_train[0], '\t')
# print(X_test[0], y_test[0])
document = '''Với sự đầu tư rót vốn từ ông lớn Netflix, Squid Game trở thành một trong 
số những phim Hàn có kinh phí sản xuất lớn nhất năm nay, lên tới khoảng 20 tỷ won (380,7 tỷ đồng). 
Dĩ nhiên, số tiền bỏ ra để trả cát xê cho các diễn viên cũng không hề nhỏ, thậm chí còn khiến khán giả phải giật mình.'''
# f = open("/content/drive/MyDrive/PhanLoaiVanBan/thunghiem.txt","r",encoding="utf8")   
# a = f.read()
document = text_preprocess(document)
print(document)
document = remove_stopwords(document)
    
nb_model = pickle.load(open(os.path.join(MODEL_PATH,"naive_bayes.pkl"), 'rb'))
label = nb_model.predict([document])
print('Predict label:', label_encoder.inverse_transform(label))
print(type(label))