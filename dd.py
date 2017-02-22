'''
ckard:
  - we have virtual beds in CVICU starting with 9999 , we exclude them
  - enter and exit time was not recored correctly! we have to clean it.
     (for more details look at ipUtilities file)

  TO Do:
  - the current version of code does not take care of imbalanced data!
  - replacing shift operators based on numbers like 24 * 3 to a time range operations look ups
'''

# for django and product purpose
import logging
from logUtils import Logger
from constants import app, event

from decision.actions.actionOption import ActionOption
from decision.actions.specializedAction import SpecializedAction
from main.decorators import method_cache
from django.db import connection
from sklearn.metrics import precision_recall_curve, average_precision_score
# required libraries
import pandas as pd
import numpy as np
from scipy import stats
import sklearn.metrics as metrics
from datetime import datetime, timedelta
from sklearn.grid_search import ParameterGrid
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# functions from other modules
import ipUtils
# import main.analysis.singleMetric as singleMetric

logger = Logger(logging.getLogger("application"), app=app.DECISION_ENGINE)

def trainTestSplits(ddParams, hospital, start, end, nfolds=1):
  '''
  returns a list of train-test data frames plus featuers for cross validation process
  we need our own version since we numerize categorical features based on target variable in the training set
  args:
    start, end : determines time range of test set, if we have more than one fole it will be the last fold
    nfolds: number of folds
  return:
    a list of train-test-features for each fold (note: it is the whole data frame not only the indices)
  '''
  trainTestFolds = [0] * nfolds
  # number of days considered in test set
  testRange = (end - start).total_seconds() / timedelta(days=1).total_seconds()
  for fold in range(nfolds):
    timeShift = timedelta(days=fold * testRange)
    trainEnd = start - timeShift
    trainStart = start - timedelta(days=ddParams['historicalDays']) - timeShift

    train = cachedTrainData(ddParams, hospital, trainStart, trainEnd)
    train, featureNames = addFeatures(ddParams, hospital, train, trainEnd)

    test = cachedTrainData(ddParams, hospital, trainEnd, trainEnd + timedelta(days=testRange))
    test, featureNames = addFeatures(ddParams, hospital, test, trainEnd)
    trainTestFolds[fold] = (train, test, featureNames)
  return trainTestFolds


def crossValidation(ddParams, hospital, model=None, trainTestFolds=None):
  '''
   a cross validation fxn for train-test splitting based on time range instead of a random split
   for now, it is only disigned to do classification problems and some specific metrics
   args:
     ddParams: DL given params
     model: the model we want to get its performance cross folds
     trainTestFolds: a list of train-test data of each folds to do cross validation 
   return:
     score: a dictionary of some evealuation metrics computed from cross validation plus below info
     trueVals_combined: a list of true values of test set
     predictions_combined: a list of score values given to test set
  '''
  
  yClasses = ddParams['targetDispositions'] + ['Other'] 
  n_classes = len(yClasses) 
  predictions_fold = [[] for i in range(n_classes)]
  trueVals_fold = [[] for i in range(n_classes)]
  auc = [0 for i in range(n_classes)]
  aupr = [0 for i in range(n_classes)]
  # Compute Precision-Recall
  precision = dict()
  recall = dict()
  average_precision = dict()
  nfolds = len(trainTestFolds)
  for fold in range(nfolds):
    train, test, featureNames = trainTestFolds[fold]
    trainX = train[featureNames]
    trainY = label_binarize(train['targetDispositions'], classes=yClasses)
    testX = test[featureNames]
    testY = label_binarize(test['targetDispositions'], classes=yClasses)
    model.fit(trainX, trainY)
    y_score = model.predict_proba(testX)
    y_test = np.array(testY)
    print 'yTest', y_test,y_test.shape
    print y_test[0,0]    
    for i in range(n_classes):
      print i
      if len(set(y_test[:, i])) == 1:
        auc[i] += 1. if y_test[0, i] == 1 else 0.
        aupr[i] += 1. if y_test[0, i] == 1 else 0.
      else:
        auc[i] += metrics.roc_auc_score(y_test[: ,i], y_score[:, i])
        aupr[i] += metrics.average_precision_score(y_test[:, i], y_score[:, i])
      auc[i] = auc[i] / float(nfolds) if nfolds > 0 else auc[i]
      aupr[i] = aupr[i] / float(nfolds) if nfolds > 0 else aupr[i]
      predictions_fold[i].extend(y_score[:, i])
      trueVals_fold[i].extend(y_test[:, i])
  
  trueVals_fold = np.array(trueVals_fold)
  predictions_fold = np.array(predictions_fold)
  for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(trueVals_fold[i], predictions_fold[i])
    average_precision[i] = average_precision_score(trueVals_fold[i], predictions_fold[i])

  # Compute micro-average ROC curve and ROC area
  precision["micro"], recall["micro"], _ = precision_recall_curve(trueVals_fold.ravel(), predictions_fold.ravel())
  average_precision["micro"] = average_precision_score(trueVals_fold, predictions_fold, average="micro")
      
  print train['targetDispositions'].value_counts()
  # to check accuracy
  score = {'auc': auc, 'aupr': aupr, 'trueVals_combined': trueVals_fold, 'predictions_combined': predictions_fold}
  return score

# to extract all relevant cases in a time range who stayed in the hospital.
def getHistoricalPatients(ddParams, hospital, start, end):
  # extra filter is an optional conditions to filtre out some cases if any
  extraFilter = '(1=1)' if ddParams['extraFilter'] is None else ddParams['extraFilter']
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(start),
    'endDatetime': str(end),
    'startSql': ddParams['startSql'],
    'endSql': ddParams['endSql'],
    'extraFilter': extraFilter
  }

  patientQuery = '''
    SELECT
      instance1,
      instance3,
      {startSql} as admit_time,
      {endSql} as discharge_time,
      TIMESTAMPDIFF(DAY, admit_time, discharge_time) as los,
      admitting_class,
      admitting_service,
      admitting_diagnosis,
      admit_source,
      age,
      gender,
      language,
      interpreter,
      primary_race,
      home_country,
      payor,
      discharge_disposition
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {startSql} is not null and
      {endSql} is not null  and
      {endSql} >= '{startDatetime}' and
      {endSql} <= '{endDatetime}' and
      {startSql} < {endSql} and
      {extraFilter}
    order by
      {startSql};
  '''.format(**queryParams)
  data = pd.read_sql_query(patientQuery, connection, coerce_float=True)
  data['admit_time'] = pd.to_datetime(data['admit_time'])
  data['discharge_time'] = pd.to_datetime(data['discharge_time'])
  return data


# --------------------------------------------------------------------------------------------------
# To calculate average of los per each group of patients
# we use this function to convert categorical features to numerical values.
# based on average LOS related to each category in the training set (before curDate)
def getGroupbyAdmit(ddParams, hospital,groupbyStr, end):
  # extra filter is an optional conditions to filtre out some cases if any
  extraFilter = '(1=1)' if ddParams['extraFilter'] is None else ddParams['extraFilter']
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(end - timedelta(days=ddParams['historicalDays'])),
    'endDatetime': str(end),
    'startSql': ddParams['startSql'],
    'endSql': ddParams['endSql'],
    'minPerBucket': ddParams['minPerBiasBucket'],
    'groupbyStr': groupbyStr,
    'extraFilter': extraFilter
  }

  query = '''
    SELECT
      {groupbyStr},
      TIMESTAMPDIFF(DAY, {startSql}, {endSql}) as los
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {startSql} is not null and
      {endSql} is not null  and
      {endSql} >= '{startDatetime}' and
      {endSql} <= '{endDatetime}' and
      {startSql} < {endSql} and
      {extraFilter}
    GROUP by
      {groupbyStr}
    HAVING
      count(*) >= {minPerBucket};
  '''.format(**queryParams)
  cursor = connection.cursor()
  cursor.execute(query)
  # Kinda complicated (done for performance). Basically makes a map from x[:-1]
  # to x[-1]. Done so we can support multiple groupbys.
  return dict([[','.join([str(z) for z in x[:-1]]), x[-1]] for x in cursor.fetchall()])


def getGroupbySurg(ddParams, hospital, groupbyStr, recDt):
  end = datetime(recDt.year, recDt.month, recDt.day)
  start = end - timedelta(days = ddParams['historicalDays'])
  caseType = ['scheduled', 'add-on']
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(start),
    'endDatetime': str(end),
    "ORstartSql": "anesthesia_ready",
    "ORendSql": "close_time",
    'caseType': tuple(caseType),
    'groupbyStr': groupbyStr,
    'minPerBucket': ddParams['minPerBiasBucket']
  }

  OR_query = '''
    SELECT
      {groupbyStr},
      AVG(
        NULLIF(TIMESTAMPDIFF(MINUTE, {ORstartSql}, {ORendSql}), 0)
      )
    FROM
      raw_or_thru
    WHERE
      hospital = {hospital} and
      {ORstartSql} >= '{startDatetime}' and
      {ORstartSql} <= '{endDatetime}' and
      {ORstartSql} is not null and
      {ORendSql} is not null and
      LOWER(case_type) IN {caseType}
    GROUP BY
      {groupbyStr}
    HAVING
      count(*) >= {minPerBucket};
  '''.format(**queryParams)
  cursor = connection.cursor()
  cursor.execute(OR_query)
  
  # Kinda complicated (done for performance). Basically makes a map from x[:-1]
  # to x[-1]. Done so we can support multiple groupbys.
  return dict([[','.join([str(z) for z in x[:-1]]), x[-1]] for x in cursor.fetchall()])
# --------------------------------------------------------------------------------------------------
# returns all patients who are in the unit in a time range and their exit time is null or after current time
#  currently there is a virtual bed starting with 9999 for patient ready to transfer in/out of unit ( we exclude them)
def getCurrentPatients(ddParams, hospital, curDate):

  queryParams = {
    'hospital': hospital.id,
    'startSql': losParams['startSql'],
    'endSql': losParams['endSql'],
    'curDate': curDate
  }

  patientQuery = '''
    SELECT
      instance1,
      instance3,
      {startSql} as admit_time,
      {endSql} as discharge_time,
      admitting_class,
      admitting_service,
      admitting_diagnosis,
      admit_source,
      age,
      gender,
      language,
      interpreter,
      primary_race,
      home_country,
      payor
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {startSql} is not null and
      ( {endSql} is null or {endSql} > '{curDate}' ) and
      {startSql} <= '{curDate}' and
    order by
      {startSql};
'''.format(**queryParams)
  data = pd.read_sql_query(patientQuery, connection, coerce_float=True)
  data['admit_time'] = pd.to_datetime(data['admit_time'])
  data['discharge_time'] = pd.to_datetime(data['discharge_time'])
  return data

def getTotalNumCases(ddParams,hospital, featureStr, end):
  # extra filter is an optional conditions to filtre out some cases if any
  extraFilter = '(1=1)' if ddParams['extraFilter'] is None else ddParams['extraFilter']
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(end - timedelta(days=ddParams['historicalDays'])),
    'endDatetime': str(end),
    'startSql': ddParams['startSql'],
    'endSql': ddParams['endSql'],
    'featureStr': featureStr,
    'extraFilter': extraFilter
  }

  query = '''
    SELECT
      {featureStr}
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {endSql} >= '{startDatetime}' and
      {endSql} <= '{endDatetime}' and
      {extraFilter};
  '''.format(**queryParams)
  data = pd.read_sql_query(query, connection, coerce_float=True)
  return data

def addSurgFeatures(ddParams, hospital, data, trainingEnd):
  # procedure name of the most recent surg proc
  procDict = getGroupbySurg(ddParams, hospital, 'procedure_name', trainingEnd)
  data['surgProc'] = [procDict.get(x['procedure_name'])
                        if x['procedure_name'] in procDict and procDict.get(x['procedure_name'])
                        else 0 for idx, x in data.iterrows()]
  roomDict = getGroupbySurg(ddParams, hospital, 'surgery_room', trainingEnd)
  data['surgRoom'] = [roomDict.get(x['surgery_room'])
                        if x['surgery_room'] in roomDict and roomDict.get(x['surgery_room'])
                        else 0 for idx, x in data.iterrows()]
  surgDict = getGroupbySurg(ddParams, hospital, 'surgeon_name', trainingEnd)
  data['surgName'] = [surgDict.get(x['surgeon_name'])
                        if x['surgeon_name'] in surgDict and surgDict.get(x['surgeon_name'])
                        else 0 for idx, x in data.iterrows()]
  asaDict = getGroupbySurg(ddParams, hospital, 'asa_class', trainingEnd)
  data['surgasa'] = [asaDict.get(x['asa_class'])
                       if x['asa_class'] in asaDict and asaDict.get(x['asa_class'])
                       else 0 for idx, x in data.iterrows()]
  # check how long last proc took compare to scheduled time
  data['overrunProcedure'] = data['overrunProcedure'].apply(lambda x: 0.0 if np.isnan(x) else round(x, 1))
  data['daysFromSurg'] = data.apply(lambda x: (x['TimeStamp'] - x['admit_time']) / timedelta(days=1)
                                      if pd.isnull(x['SurgeryStop'])
                                      else (x['TimeStamp'] - x['SurgeryStop']) / timedelta(days=1), axis=1)
  # need another feature to distinguish between those had surgeries from those do not have
  data['hadSurg'] = data['SurgeryStop'].apply(lambda x: 0 if pd.isnull(x) else 1)
  surgFeatures = ['surgProc', 'surgRoom', 'surgName', 'surgasa', 'overrunProcedure', 'daysFromSurg', 'hadSurg']
  return data, surgFeatures


def addCategoricalAdmitFeat(ddParams, hospital, data, trainingEnd):
  data['gender'] = data['gender'].apply(lambda x: 1 if x == 'M' else 0)
  data['interpreter'] = data['interpreter'].apply(lambda x: 1 if x == 'Y' else 0)
  features = ['gender', 'interpreter'] 

  def findTopcommon(ddParams, hospital, colName, trainingEnd):
    historicalValues = getTotalNumCases(ddParams, hospital, colName, trainingEnd)
    frequencies = historicalValues[colName].value_counts()
    return frequencies[frequencies/historicalValues.shape[0] >= 0.01].index.tolist()

  topLanguages = findTopcommon(ddParams, hospital, 'language', trainingEnd)
  data, languageFeatures = ipUtils.oneHotEncoder(data, colName='language', Categories=topLanguages)
  features += languageFeatures
  topRaces = findTopcommon(ddParams, hospital, 'primary_race', trainingEnd)
  data, raceFeatures = ipUtils.oneHotEncoder(data, colName='primary_race', Categories=topRaces)
  features += raceFeatures
  topCountries = findTopcommon(ddParams, hospital, 'home_country', trainingEnd)
  data, countryFeatures = ipUtils.oneHotEncoder(data, colName='home_country', Categories=topCountries)
  features += countryFeatures
  topPayors = findTopcommon(ddParams, hospital, 'payor', trainingEnd)
  data, payorFeatures = ipUtils.oneHotEncoder(data, colName='payor', Categories=topPayors)
  features += payorFeatures

  # for below features we numerize categories based on thier historical average los 
  admitServiceMap = getGroupbyAdmit(ddParams, hospital, 'admitting_service', trainingEnd)
  data['admitService'] = [admitServiceMap.get(x['admitting_service'])
                      if x['admitting_service'] in admitServiceMap and admitServiceMap.get(x['admitting_service'])
                      else 0 for idx, x in data.iterrows()]
  admitClassMap = getGroupbyAdmit(ddParams, hospital,'admitting_class', trainingEnd)
  data['admitClass'] = [admitClassMap.get(x['admitting_class'])
                      if x['admitting_class'] in admitClassMap and admitClassMap.get(x['admitting_class'])
                      else 0 for idx, x in data.iterrows()]
  admitDiagnosisMap = getGroupbyAdmit(ddParams, hospital,'admitting_diagnosis', trainingEnd)
  data['admitDiagnosis'] = [admitDiagnosisMap.get(x['admitting_diagnosis'])
                      if x['admitting_diagnosis'] in admitDiagnosisMap and admitDiagnosisMap.get(x['admitting_diagnosis'])
                      else 0 for idx, x in data.iterrows()]
  admitSourceMap = getGroupbyAdmit(ddParams,hospital, 'admit_source',trainingEnd)
  data['admitSource'] = [admitSourceMap.get(x['admit_source'], trainingEnd)
                      if x['admit_source'] in admitSourceMap and admitSourceMap.get(x['admit_source'])
                      else 0 for idx, x in data.iterrows()]
  features += ['admitService', 'admitClass', 'admitDiagnosis', 'admitSource']

  return data, features


def addFeatures(ddParams, hospital, data, end):
  '''
  start, end: start and end time of input census data we tend to add new features
  patientData: a data frame of patient information used to build census data
  patLookup: a class storing all historical average of los for different categorial features
  '''
  data['month'] = data['TimeStamp'].apply(lambda x: x.month)
  data['year'] = data['TimeStamp'].apply(lambda x: x.year)
  data['season'] = data['TimeStamp'].apply(lambda x: ipUtils.get_season(x))
  features = ['month', 'year', 'season']
  
  data, admitFeatures = addCategoricalAdmitFeat(ddParams, hospital, data, end)
  features += admitFeatures
  
  data, surgFeatures = addSurgFeatures(ddParams, hospital, data, end)
  features += surgFeatures

  # if patient is re-admitted
  allCases = getTotalNumCases(ddParams,hospital, 'instance1, admit_time', end)
  data['re_admitted'] = [0 if allCases[(allCases.instance1 == row.instance1) & (allCases.admit_time < row.admit_time)].empty else 1 for idx,row in data.iterrows()]
  # current los of patient
  data['currentLOS'] = data.apply(lambda x: (x['TimeStamp'] - x['admit_time'])/timedelta(days=1), axis=1)
  features += ['re_admitted', 'currentLOS']

  ageFeatureNames = ipUtils.addAgeFeatures(data)
  features += ageFeatureNames 

  # we defi
  print features
  return data, features

def cachedTrainData(ddParams, hospital, start, end):
  '''
  a function to return data and features for training and cross validation
  '''
  raw_data = getHistoricalPatients(ddParams, hospital, start, end)
  raw_data['targetDispositions'] = raw_data['discharge_disposition'].apply(lambda x: x if x in ddParams['targetDispositions'] else 'Other')
  # we blow up data to have time stamp for patients during thier stay
  patientsData =ipUtils.createTimeStampedData(raw_data,enterColumn='admit_time',exitColumn='discharge_time')
  # we do not wish to trigger on the same day as admit_time or discharge time
  patientsData = patientsData[(patientsData.TimeStamp != patientsData.admit_time) &
                              (patientsData.TimeStamp != patientsData.discharge_time)]
  # we add OR features if patients had surgeries
  patientsData = ipUtils.addORColumns(data=patientsData, losParams=ddParams, hospital=hospital, unitFilter=False)
  return patientsData


@method_cache(60 * 60 * 12)
def cachedTrainModel(ddParams, hospital, start, end):
  patientsData = cachedTrainData(ddParams, hospital, start, end)
  data, featureNames = addFeatures(ddParams, hospital, patientsData, end)
  # Binarize the outpu
  trainX = data[featureNames]
  if len(trainX) <= 0:
    raise ValueError("No training data for discharge disposition prediction!")
  if ddParams['model']:
    model = ipUtils.getModel(ddParams['model'], ddParams['modelParams'])
    probabilityThreshold = 0.5
  
  model.fit(trainX, trainY) 

  logger.info(event=event.IP_DISCHARGE_DISPOSITION_PREDICTION_LOOP,
              msg='Discharge Disposition Prediction training data size',
              username='admin',
              sample_size=len(trainY))

  return model, featureNames


def trainModel(ddParams, hospital, recDt):
  trainingStart = recDt - timedelta(days=ddParams['historicalDays'])
  trainingEnd = recDt
  model, featureNames = cachedTrainModel(ddParams, hospital, trainingStart, trainingEnd)
  return model, featureNames


def testModel(ddParams, hospital, featureNames, model, curDate):
  currentpatients = getCurrentPatients(ddParams, hospital, curDate)
  data, featureNames = addFeatures(ddParams, hospital, currentpatients, end=curDate - timedelta(days=1))

  testX = data[featureNames]
  # We already pass in model.
  pred = model.predict_proba(testX) if len(testX) > 0 else []
  ret = []
  for instance3, unit, bed, pred in \
      zip(data['instance3'], data['unit'], data['bed'], preds):
        ret.append([instance3, unit, bed, pred ])

  return ret


# ----------------------------------------------------------------------------------
class DischargeDispositionPrediction(SpecializedAction):
  """
  Predicting if number of patients in a unit (CVICU in this case) few days ahead (default 3 days)
  at/over a thershold(default 19) as a full, otherwise not full
  """

  VERSION = '1.0.0'
  TRAINING_FREQUENCY = timedelta(days=7)
  NEEDS_ASYNC_TRAINING = True
  '''
  numDaysAhead: number of days ahead we will do prediction , the default is end of september 2014
  maxCapWindow : max number of patients cross this window of time will be used as target variable
  triggerTime : hour of day we pan to trigger

  '''
  defaultParameters = {
    'historicalDays': 100,
    'triggerTime': 7,
    'recallThreshold': None,
    'startSql': 'admit_time',
    'endSql': 'discharge_time',
    'targetDispositions':[
      'Acute Inpatient Hospital',
      'Skilled Nursing Facility/Intermediate Facility',
      'Hospice-Medical Facility',
      'Inpatient Rehab',
      'Residential Care Facility',
      'Long Term Care/Subacute Facility'
    ],
    'extraFilter': None,
    'template': ''' 
                  Our model predicts the disposition of patient currently on the bed {bed} in the {unit}
                  after discharge would be {pred}. We are awesome ;)
                ''',
    'personalized': False,
    # Not in the wiki. Largely for testing.
    'model': 'Randomforestclassifier',
    'modelParams': {},
    'minPerBiasBucket': 4
  }

  def generateTestingData(self, userProfile, action, start, end, nfolds=1):
    '''
     nfolds is used when we like to test on many folds
     start and end determines start and end time of the first test set. The rest of test sets will be
     accordingly adjusted.
    '''
    start = datetime(start.year, start.month, start.day, start.hour)
    end = datetime(end.year, end.month, end.day, end.hour)
    defaultParams = DischargeDispositionPrediction.defaultParameters
    ddParams = self.processParameters(defaultParams, action.parameters)
    hospital = action.hospital

    if ddParams['model']:
      model = ipUtils.getModel(ddParams['model'], ddParams['modelParams'])
      classifier = OneVsRestClassifier(model)
      trainTestFolds = trainTestSplits(ddParams, hospital, start, end, nfolds)
      score = crossValidation(ddParams, hospital, classifier, trainTestFolds)

    ret = ['score', score]
    return ret

  def trainModel(self, action, recDt=None):
    if recDt is None:
        recDt = datetime.today()
    recDt = datetime(recDt.year, recDt.month, recDt.day, recDt.hour)
    defaultParams = DischargeDispositionPrediction.defaultParameters
    ddParams = self.processParameters(defaultParams, action.parameters)
    model, featureNames = trainModel(ddParams, action.hospital, recDt)

    return {'featureNames': featureNames, 'model': model}

  def getNotificationTemplate(self, ddParams):
    ret = ddParams['template']
    return ret
'''
  def getOptions(self, userProfile, action, execDt, recDt):
    originalRecDt = datetime(recDt.year, recDt.month, recDt.day, recDt.hour, recDt.minute, recDt.second)

    defaultParams = DischargeDispositionPrediction.defaultParameters
    ddParams = self.processParameters(defaultParams, action.parameters)

    # Check the db to see if there is a fitted model there.
    modelParams = action.getMostRecentFittedModel(self)
    if modelParams is None:
      return []

    model = modelParams['model']
    featureNames = modelParams['featureNames']
    results = testModel(ddParams, action.hospital, featureNames, model, recDt)
    ret = []
    for instance3, unit, bed, prediction in results:
      predictionTime = datetime(recDt.year, recDt.month, recDt.day, recDt.hour)
      ddParams['targetDispositions'][]
      if not ddParams['personalized'] and recDt.hour == ddParams['triggerTime'] and prediction == 1:
        templateVars = {
          'unit': unit,
          'bed': bed
          'pred': prediction
        }
        notifTemplate = self.getNotificationTemplate(ddParams)
        actionString = notifTemplate.format(**templateVars)
        option = ActionOption(action, actionString, 100.0, originalRecDt)
        # Duration because they might change scheduled time tomorrow, when this class might still run.
        option.setUniqueId('%s' % str(predictionTime))
        # option.updateMetaData({'prediction': str(prediction)})
        ret.append(option)

        logger.info(event=event.IP_CAPACITY_FULL_PREDICTION_DECISION_LOOP,
                    msg='Capacity Full Prediction trigger information',
                    username='admin',
                    predictionTime=predictionTime,
                    unit=ddParams['unit'],
                    probabilityThreshold=probabilityThreshold,
                    rec_dt=originalRecDt,
                    prediction=prediction,
                    model=model)
      else:
        logger.info(event=event.IP_CAPACITY_FULL_PREDICTION_DECISION_LOOP,
                    msg='Capacity Full Prediction no trigger',
                    username='admin',
                    predictionTime=predictionTime,
                    unit=ddParams['unit'],
                    probabilityThreshold=probabilityThreshold,
                    rec_dt=originalRecDt,
                    prediction=prediction,
                    model=model)
    return ret
'''
