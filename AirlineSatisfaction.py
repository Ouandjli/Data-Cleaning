import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import data as dt
import matplotlib.ticker as mtick
from sklearn.tree import DecisionTreeClassifier


FileName='AirlineSatisfaction_2.csv'
df=pd.read_csv(FileName)
print(df.head())

#Convertire les valeurs
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
df['Customer Type'] = df['Customer Type'].replace({'Loyal Customer': 1, 'disloyal Customer': 0})
df['Type of Travel'] = df['Type of Travel'].replace({'Business travel': 1,'Personal Travel':0})
df['Class'] = df['Class'].replace({'Eco': 0, 'Eco Plus': 1,'Business': 2})
df['satisfaction'] = df['satisfaction'].replace({'satisfied': 1, 'neutral or dissatisfied': 0})
print(df.head())


#donnèes manquantes
#Forward Methode

df_filled = df.fillna(df.median())
df_filled.to_csv('AirlineSatisfaction_2.csv')
df_filled['Customer Type'] = df_filled['Customer Type'].astype(int)
Y = df_filled['satisfaction']
X = df_filled[['Gender','Customer Type','Age','Type of Travel','Class','Flight Distance','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding',
        'Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Departure Delay in Minutes','Arrival Delay in Minutes']]

X = sm.add_constant(X)
# Fit logistic regression model
model = sm.Logit(Y,X)
result = model.fit()
print(result.summary())
#Question2, methode BACKWARD
Y = df_filled['satisfaction']
X = df_filled[['Gender','Customer Type','Age','Type of Travel','Class','Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Online boarding','On-board service','Leg room service','Baggage handling','Checkin service','Inflight service','Cleanliness','Arrival Delay in Minutes']]
X = sm.add_constant(X)
model = sm.Logit(Y,X)
result = model.fit()
print(result.summary())


#méthode Backward
def backward_elimination_logistic(X, y, threshold=0.05):
    num_vars = X.shape[1]
    for i in range(num_vars):
        model = sm.Logit(y, X).fit()
        max_pvalue = max(model.pvalues)  # Obtenir la p-value maximale
        if max_pvalue > threshold:
            max_index = model.pvalues.idxmax()  # Obtenir l'indice de la variable avec la p-value maximale
            X.drop(columns=[max_index], inplace=True)  # Retirer la variable correspondante de X
        else:
            break
    print(result.summary())


# question 3 sensitivities to variables
Y = df_filled['satisfaction']
X = df_filled[['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Inflight wifi service',
               'Departure/Arrival time convenient', 'Ease of Online booking', 'Online boarding', 'On-board service',
               'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
               'Arrival Delay in Minutes']]
X = sm.add_constant(X)
model = sm.Logit(Y, X)
result = model.fit()
beta = result.params
T = len(Y)
# X.iloc[:, 1]
Y_hat = 1. / (1 + np.exp(-(
            beta[0] * X['const'] + beta[1] * X['Customer Type'] + beta[2] * X['Type of Travel'] + beta[3] * X[
        'Inflight wifi service'] + beta[4] * X['Online boarding']
            + beta[5] * X['On-board service'] + beta[6] * X['Leg room service'] + beta[7] * X['Checkin service'])))
Y_hat_dummy = np.zeros(T)
Y_hat_dummy[Y_hat >= 0.5] = 1

## Fonction pour calculer les valeurs prédites Y_hat
def calculate_Y_hat(beta, X):
    #Initialisation du produit scalaire avec l'intercept
    linear_predictor = beta[0] * X['const']
    for i in range(1, len(beta)):
        linear_predictor += beta[i] * X.iloc[:, i]
    yhat = 1. / (1 + np.exp(-linear_predictor))
    return yhat

Y_hat = calculate_Y_hat(beta,X)


# stats -------------------------------------------------------------------------
status = {}
for s in range(2):
    status['reality.' + str(s)] = np.sum(Y == s)
    status['model.' + str(s)] = np.sum(Y_hat_dummy == s)
print("====================================")
print("STATS:")
print("====================================")
print("satisfaction\t reality\t model 1")
for s in range(2):
    print("%d\t\t\t %d\t\t\t %d"%(s,status['reality.' + str(s)] ,status['model.' + str(s)]))
print("Total\t\t %d\t\t\t%d"%(status['reality.0'] + status['reality.1'],status['model.0'] + status['model.1']))
HitRatio = np.sum(Y_hat_dummy == Y.values) / T
print("HitRatio = %1.1f%%"%(100 * HitRatio))
FalsePositives = np.sum((Y.values == 0) & (Y_hat_dummy == 1)) / T
FalseNegatives = np.sum((Y.values == 1) & (Y_hat_dummy == 0)) / T
print("False positives = %1.1f%%"%(100 * FalsePositives))
print("False negatives = %1.1f%%"%(100 * FalseNegatives))

# trouver les FN/FP customer type

Y = df_filled['satisfaction']
X = df_filled['Customer Type']
X = sm.add_constant(X)
model = sm.Logit(Y, X)
result = model.fit()
beta = result.params
T = len(Y)
# X.iloc[:, 1]
Y_hat = 1. / (1 + np.exp(-(beta[0] * X['const'] + beta[1] * X['Customer Type'])))
Y_hat_dummy = np.zeros(T)
Y_hat_dummy[Y_hat >= 0.5] = 1
status = {}
for s in range(2):
    status['reality.' + str(s)] = np.sum(Y == s)
    status['model.' + str(s)] = np.sum(Y_hat_dummy == s)
print("====================================")
print("STATS:")
print("====================================")
print("satisfaction\t reality\t model 1")
for s in range(2):
    print("%d\t\t\t %d\t\t\t %d"%(s,status['reality.' + str(s)] ,status['model.' + str(s)]))
print("Total\t\t %d\t\t\t%d"%(status['reality.0'] + status['reality.1'],status['model.0'] + status['model.1']))
HitRatio = np.sum(Y_hat_dummy == Y.values) / T
print("HitRatio = %1.1f%%"%(100 * HitRatio))
FalsePositives = np.sum((Y.values == 0) & (Y_hat_dummy == 1)) / T
FalseNegatives = np.sum((Y.values == 1) & (Y_hat_dummy == 0)) / T
print("False positives = %1.1f%%"%(100 * FalsePositives))
print("False negatives = %1.1f%%"%(100 * FalseNegatives))


#PLT âge
bins = [0, 20, 60, float('inf')]
labels = ['0-20', '20-60', '60+']
age_counts = {'0-20': 150, '20-60': 300, '60+': 100}
labels = age_counts.keys()
heights = age_counts.values()
plt.bar(labels, heights, color='skyblue')

plt.xlabel('Groupe d\'âge')
plt.ylabel('Nombre d\'observations')
plt.title('Répartition des âges par groupe')
plt.figure()

plt.show()
#âge et leurs satisfaction
bins = [0, 20, 60, float('inf')]
labels = ['0-20', '20-60', '60+']
data = data.assign(age_group=pd.cut(data['âge'], bins=bins, labels=labels, right=False))
satisfaction_by_age = data.groupby('age_group')['satisfaction'].mean()
x = satisfaction_by_age.index.astype(str)  # Convertir les étiquettes en chaînes de caractères
y = satisfaction_by_age.values
plt.scatter(x, y, color='skyblue')
plt.xlabel('Groupe d\'âge')
plt.ylabel('Satisfaction moyenne')
plt.title('Satisfaction moyenne par groupe d\'âge')
plt.show()
