from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# Thống kê số lượng data theo nhãn
# count = {}
# for line in open('news_categories.txt',encoding='utf-8'):
#     key = line.split()[0] #tách câu ra thành một list các từ
#     count[key] = count.get(key, 0) + 1

# for key in count:
#     print(key, count[key])

# Thống kê các word xuất hiện ở tất cả các nhãn
total_label = 18
vocab = {}
label_vocab = {}
for line in open('news_categories.txt',encoding='utf-8'):
    words = line.split()
    # lưu ý từ đầu tiên là nhãn
    label = words[0]
    if label not in label_vocab:
        label_vocab[label] = {}
    for word in words[1:]:
        label_vocab[label][word] = label_vocab[label].get(word, 0) + 1
        if word not in vocab:
            vocab[word] = set()
        vocab[word].add(label)

count = {}
for word in vocab:
    if len(vocab[word]) == total_label:
        count[word] = min([label_vocab[x][word] for x in label_vocab])
        
sorted_count = sorted(count, key=count.get, reverse=True)
for word in sorted_count[:100]:
    print(word, count[word])

stopword = set()
with open('stopwords.txt', 'w',encoding='utf-8') as fp:
    for word in sorted_count[:100]:
        stopword.add(word)
        fp.write(word + '\n')
    
def remove_stopwords(line):
    words = []
    for word in line.strip().split():
        if word not in stopword:
            words.append(word)
    return ' '.join(words)
    
    
with open('news_categories.prep', 'w',encoding='utf-8') as fp:
    for line in open('news_categories.txt',encoding='utf-8'):
        line = remove_stopwords(line)
        fp.write(line + '\n')

# Chia tập train/test
test_percent = 0.2

text = []
label = []

for line in open('news_categories.prep',encoding='utf-8'):
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

print(X_train[0], y_train[0], '\t')
print(X_test[0], y_test[0])

MODEL_PATH = "models"
import os
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

start_time = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), #vector hóa cho từng từ
                                             max_df=0.8, #loại bỏ các từ xuất hiện trên 80% trong bộ ký tự
                                             max_features=None)), #sẽ chọn các từ xuất hiện thường xuyên nhất trong vocab
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)) #Tham số làm mịn, fit trên dữ liệu train để tính P(B)
                    ])
text_clf = text_clf.fit(X_train, y_train)

train_time = time.time() - start_time
print('Done training Naive Bayes in', train_time, 'seconds.')

# Save model
pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'wb'))