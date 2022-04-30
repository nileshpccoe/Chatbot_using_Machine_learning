import numpy as np 
import string

import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from textblob import TextBlob


df = pd.read_csv('dialogs.txt',sep='\t')
df


df = pd.read_csv('dialogs.txt',sep='\t')
df = df.rename(columns={"hi, how are you doing?":"Questions","i'm fine. how about yourself?":"Answers"})

df3 = pd.DataFrame([['hi, how are you doing?','im fine. how about yourself?'],
                   ['Hi','hello'],
                   ['Hello','hi'],
                   ['how are you',"i'm fine. how about yourself?"],
                   ['how are you doing',"i'm fine. how about yourself?"],
                   ['what is good name','it me aarush']],
                   columns=['Questions','Answers'])
df=pd.concat([df,df3],sort=False)
df


import string
string.punctuation


list_new=[]
def clean_text(text):
    a=[char for char in text if char not in string.punctuation ]
    text=''.join(a)
    text=[word.lower() for word in text.split() if word.lower()]
    for i in text:
        msg=TextBlob(i)
        i=msg.correct()
        list_new.append(str(i))
        text=list_new
            
            
    return text


x=df['Questions']

from sklearn.feature_extraction.text import TfidfVectorizer


tfidf=TfidfVectorizer(analyzer=clean_text)


x_trans=tfidf.fit_transform(x)

x_vect=pd.DataFrame(x_trans.toarray())

x_vect.shape

y=df['Answers']

y.shape



from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10,random_state=0)




Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=clean_text)),
    ('tfidf',TfidfTransformer()),
    ('classifier',clf)
])
Pipe.fit(df['Questions'],df['Answers'])


def get_output(text_msg):
    output=Pipe.predict([text_msg])[0]
    return output

print(Pipe.predict(['hi'])[0])







