from main import LymeDisease
from sklearn.linear_model import LogisticRegression

l = LymeDisease()

model = LogisticRegression(max_iter=10000).fit(l.x_train, l.y_train)
acc = model.score(l.x_valid, l.y_valid)*100
print(f'The model accuracy is {acc:.2f}%')
"""
Accuracy is pretty low and we desire an accuracy at least in the 80 percent range. 
"""
