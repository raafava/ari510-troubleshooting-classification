"""Train reproducible troubleshooting classifiers; never manually label test data."""
import argparse,json,random,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
class TextStats(BaseEstimator,TransformerMixin):
 def fit(self,X,y=None):return self
 def transform(self,X):
  features=[]
  for t in X:
   words=re.findall(r"\b\w+\b",t.lower());n=len(words)
   features.append([np.log1p(len(t)),np.log1p(n),np.log1p(t.count('\n')),len(set(words))/max(1,n),len(re.findall(r"\d+",t)),len(re.findall(r"https?://",t)),len(re.findall(r"\b(error|failed|failure|code|message)\b",t.lower())),len(re.findall(r"\b(tried|restarted|reinstalled|checked|tested|updated|reset|rebooted)\b",t.lower())),len(re.findall(r"\b(windows|linux|mac|android|ios|ubuntu|version|ram|cpu|gpu)\b",t.lower())),t.count('?'),len(re.findall(r"\b(when|after|before|since|during)\b",t.lower())),np.mean([len(w) for w in words]) if words else 0])
  return np.asarray(features,dtype=float)
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args();d=Path(a.data);o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
np.random.seed(42);random.seed(42)
tr=pd.read_csv(d/'train.csv').fillna('');te=pd.read_csv(d/'test.csv').fillna('');sample=pd.read_csv(d/'sample_submission.csv')
assert tr.id.is_unique and te.id.is_unique and set(tr.id).isdisjoint(te.id)
assert set(tr.sufficiency)=={0,1};assert len(te)==50
X=tr.text;y=tr.sufficiency.astype(int).to_numpy();cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
configs=[
 ('run1','Word TF-IDF + balanced logistic regression',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=3000),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42)),
 ('run2','Character TF-IDF + balanced logistic regression',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=5000),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=42)),
 ('run3','Word TF-IDF + trained 16-unit neural network',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=1500),MLPClassifier(hidden_layer_sizes=(16,),activation='relu',solver='lbfgs',alpha=1,max_iter=1500,random_state=42)),
 ('run4','Character TF-IDF + trained 16-unit neural network',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=3000),MLPClassifier(hidden_layer_sizes=(16,),activation='relu',solver='lbfgs',alpha=1,max_iter=1500,random_state=42))]
configs.extend([
 ('run5','Text detail statistics + balanced logistic regression',Pipeline([('stats',TextStats()),('scale',StandardScaler())]),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000,random_state=42)),
 ('run6','Text detail statistics + trained 8-unit neural network',Pipeline([('stats',TextStats()),('scale',StandardScaler())]),MLPClassifier(hidden_layer_sizes=(8,),solver='lbfgs',alpha=5,max_iter=1500,random_state=42))])
results=[];oof=pd.DataFrame({'id':tr.id,'true_label':y})
for run,name,v,clf in configs:
 model=Pipeline([('tfidf',v),('classifier',clf)])
 print('Cross-validating',run,name,flush=True)
 # TF-IDF vocabulary and IDF are fitted only on training folds.
 prob=cross_val_predict(model,X,y,cv=cv,method='predict_proba',n_jobs=1)[:,1]
 thresholds=np.arange(.2,.701,.05)
 score,_,threshold=max((f1_score(y,prob>=t,zero_division=0),-abs(float(t)-.5),float(t)) for t in thresholds)
 pred=(prob>=threshold).astype(int)
 model.fit(X,y);testprob=model.predict_proba(te.text)[:,1];labels=(testprob>=threshold).astype(int)
 lookup=dict(zip(te.id,labels));submission=pd.DataFrame({'id':sample.id,'sufficiency':[int(lookup[k]) for k in sample.id]})
 assert len(submission)==50 and submission.id.is_unique and set(submission.sufficiency)<= {0,1}
 submission.to_csv(o/(run+'_submission.csv'),index=False);joblib.dump(model,o/(run+'_model.joblib'))
 entry={'run_id':run,'approach':name,'cv_f1_at_0_5':f1_score(y,prob>=.5,zero_division=0),'cv_f1_tuned_threshold':score,'threshold':threshold,'cv_precision':precision_score(y,pred,zero_division=0),'cv_recall':recall_score(y,pred,zero_division=0),'confusion_matrix':confusion_matrix(y,pred).tolist(),'public_leaderboard_f1':None,'seed':42,'folds':4,'model_settings':clf.get_params(),'vectorizer_settings':{k:str(val) for k,val in v.get_params().items()}}
 results.append(entry);oof[run+'_probability']=prob
 print(json.dumps({k:entry[k] for k in ['run_id','cv_f1_at_0_5','cv_f1_tuned_threshold','threshold','cv_precision','cv_recall']}),flush=True)
(o/'results.json').write_text(json.dumps(results,indent=2));oof.to_csv(o/'cv_predictions.csv',index=False)
best=max(results,key=lambda r:r['cv_f1_tuned_threshold'])
import shutil
shutil.copyfile(o/(best['run_id']+'_submission.csv'),o/'best_submission.csv')
print('BEST',best['run_id'],best['cv_f1_tuned_threshold'],flush=True)
(o/'data_hashes.json').write_text(json.dumps({f:hashlib.sha256((d/f).read_bytes()).hexdigest() for f in ['train.csv','test.csv','sample_submission.csv']},indent=2))
