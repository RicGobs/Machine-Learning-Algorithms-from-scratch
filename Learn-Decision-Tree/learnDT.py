#IMPLEMENTAZIONE SCRATCH
#importo librerie necessarie e ottengo i dati
import pandas as pd
import numpy as np
from pprint import pprint

#import the dataset
dataset = pd.read_csv("restaurant_waiting.csv",names=['Alt','Bar','Fri','Hun','Pat','Price','Rain','Res','Type','Est','Wait'])

#entropy function
#target=column 'Wait'
def entropia(target):
    #calcola valore e conta per gli attributi_split
    elementi,conta = np.unique(target,return_counts=True) 
    #return_counts=True -> ritorna il numero delle volte in cui ci sta target
    entropy = np.sum([(-conta[i]/np.sum(conta))*np.log2(conta[i]/np.sum(conta)) for i in range(len(elementi))]) #funzione di entropia 
    return entropy

#function importance (information gain)
#data=nostro dataset
#attributi che usciamo per calcolare importance
#target=sempre target 'Wait'
def Importance(data,attributi_split,target="Wait"): 
    val,counts = np.unique(data[attributi_split],return_counts=True)
    #chiamata a entropia per capire l'importanza
    diff = np.sum([(counts[i]/np.sum(counts))*entropia(data.where(data[attributi_split]==val[i]).
                                dropna()[target])for i in range(len(val))])
                                #uso dropna per togliere valori null,  doppioni infatti non sono utili 
    #formula for importance
    tot = entropia(data[target])
    info = tot-diff
    return info
    
def LearnDT(data,examples,padre): 
    #examples e padre sono alla prima sono uguali, poi cambiano
    #tutti target_values hanno stesso valore allora ritorno il valore
    a='Wait'
    if len(np.unique(data[a])) <= 1:
        return np.unique(data[a])[0] 
               #trova elementi univoci dell'array 
    #se dataset vuoto allora ritorno target iniziale
    elif len(data) == 0:           
    #plurality_value la faccio con la funzione di numpy -> ottengo un array ordinato di elementi unici
        return np.unique(dataset[a])[np.argmax(np.unique(dataset[a],   #doppioni infatti non sono utili 
                                                                           return_counts=True)[1])]  
    #se l'insieme delle features è vuoto allora ritorno il padre, il nodo precedente
    elif len(examples) == 0:
        return padre 
    #vado sull'albero
    else: 
        #metto valore per sto nodo dell'albero
        padre = np.unique(data[a])[np.argmax(np.unique(data[a],
                                                                           return_counts=True)[1])]
    #scelgo con importance con quale attributo lavorare
    elementi = [Importance(data,e,a)for e in examples] 
    j = np.argmax(elementi)
    example_scelto = examples[j]


    #stampa albero
    tree = {example_scelto:{}}
    #tolgo l'example scelto
    elem = [i for i in examples if i!= example_scelto]
    #albero
    for value in np.unique(data[example_scelto]):
        value = value
        sub_data = data.where(data[example_scelto]==value).dropna() 
        #forse basta drop() visto che so che non ci stanno null
        sottotree = LearnDT(sub_data,elem,padre) 
        #ricorsione
        tree[example_scelto][value] = sottotree
    return(tree)           
    
    
#Predizione delle scelte, date la tupla lui risponde si o no
def predizione(q,albero,default=1):
    for i in list(q.keys()):   
        #stampa si/no in base ai nodi foglia
        if i in list(albero.keys()):
            try: #necessario o da errore
               result = albero[i][q[i]] 
               #albero come dizionario
            except:
               return default

            result = albero[i][q[i]]
            if isinstance(result,dict): 
                return predizione(q,result)
            else:
                return result  
                
def split(dataset): 
    #controlla che la predizione sia fatta bene
    training_data = dataset.iloc[3:13].reset_index(drop=True)
    testing_data = dataset.iloc[1:3].reset_index(drop=True) 
    #un solo valore per test
    return training_data,testing_data
training_data = split(dataset)[0]

testing_data = split(dataset)[1]

def verifica(data,tree):   
   queries = data.iloc[:,:-1].to_dict(orient="records")
   predicted = pd.DataFrame(columns=["predicted"])
   for i in range(len(data)):
       predicted.loc[i,"predicted"] = predizione(queries[i],tree,1.0)
   print("The Prediction accuracy is:",(np.sum(predicted["predicted"]==data["Wait"])/len(data))*100,'%') 
                                                     #funzione: somma delle predizioni che mi portano a wait / lunghezza dati di verifica (in percentuale)
#stampa di controllo
tree = LearnDT(training_data,training_data.columns[:-1],training_data.columns[:-1])
pprint(tree)
verifica(testing_data,tree)     














#IMPLEMENTAZIONE CON SKLEARN
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#dopo le librerie, sistemo i dati poichè sklearn ama lavorare coi numeri
#riordino poi la tabella, altrimenti prendo gli attributi sbagliati

data = pd.read_csv("restaurant_waiting.csv")
data=data.replace(to_replace ="Yes",
                 value ="1")
data=data.replace(to_replace ="No",
                 value ="0")
pat_data=LabelEncoder()
price_data=LabelEncoder()
type_data=LabelEncoder()
est_data=LabelEncoder()
data['pat_data']=pat_data.fit_transform(data['Pat'])
data['price_data']=pat_data.fit_transform(data['Price'])
data['type_data']=pat_data.fit_transform(data['Type'])
data['est_data']=pat_data.fit_transform(data['Est'])
data=data.drop(['Pat','Price','Type','Est'],axis='columns')
data = data.reindex(columns=['Alt',	'Bar',	'Fri',	'Hun',	'pat_data',	'price_data', 'Rain',	'Res',	'type_data',	'est_data',	'Wait'])

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from sklearn import tree

#riscrivo librerie
#cerco accuratezza con sklearn
X=data.iloc[:,1:10]
y=data.iloc[:,10] #fornisco stesse righe                      #percentuale simile alla mia 
x_train, x_test, y_train, y_test = train_test_split(X,y,train_size=0.8,random_state=42)
clf=DecisionTreeClassifier(criterion='entropy',random_state=1)
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
score=accuracy_score(y_pred,y_test)
print()
print(score)

#decommentare se si vuole vedere la stampa del tree con sklearn
#scaricare e importare matplotlib 

#tree.plot_tree(clf);
#ha la stessa struttura del mio e un po' più di accuratezza, bene, ha senso
              
