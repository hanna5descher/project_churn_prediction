import pickle
import pandas as pd

def modelRF(dfFeatures):
    filename = 'hanna5descher/random_forest_SMOTE_randomseed10.sav'
    model = pickle.load(open(filename, 'rb'))
    #print("Loaded random forest model:", filename)

    colFeatures = dfFeatures.columns[3:]
    X = dfFeatures[colFeatures]

    predY = model.predict(X)  # binary classification: based on 0.5 threshold
    probY = model.predict_proba(X)  # probability

    dfModel = pd.DataFrame(columns=['userid', 'start_service', 'location', 'churn_prob', 'risk'])
    dfModel['userid'] = dfFeatures['userid']
    dfModel['start_service'] = dfFeatures['start_service']
    dfModel['location'] = dfFeatures['stateID']
    dfModel['churn_prob'] = probY[:, 1]
    dfModel['risk'] = predY

    dfModel.sort_values(by='churn_prob', ascending=False, inplace=True)

    dfModel.round({'churn_prob': 2})
    dictRisk = {1: 'High', 0: 'Low'}
    dfModel['risk'] = dfModel['risk'].map(dictRisk)

    return dfModel