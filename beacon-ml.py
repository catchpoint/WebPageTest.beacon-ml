#!/usr/bin/python
"""
Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
from data_utils import compute_variable_histograms
from data_utils import check_session_uniqueness
from data_utils import subsample_and_vectorize_data
import numpy as np
import json

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import sys

RUN_CHECKS = False
REGEN_DATA = False
# one of deep_model, logistic_regression, random_forest, None
#TRAIN = 'deep_model'
TRAIN = 'random_forest'

# Training bounce or conversion
#LABEL = 'is_converted'
#PRETTY_PRINT_LABEL = 'conversion'
LABEL = 'is_bounce'
PRETTY_PRINT_LABEL = 'bounce'

# Including bounce rate data
DATA_DIR = './data'
RESULTS_DIR = './results'
CSV_FNAME = 'beacon.csv'

VECTOR_DATA_PATH = DATA_DIR + '/' + PRETTY_PRINT_LABEL + '_data.npy'
VECTOR_LABELS_PATH = DATA_DIR + '/' + PRETTY_PRINT_LABEL + '_labels.npy'
MODEL_PATH = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_model'
VALUE_RANGES_PATH = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_value_ranges.json'
IMPORTANCES_CSV = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_importances.csv'
DATA_CSV = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_data_'

EPOCH_COUNT = 200
FOREST_SIZE = 1000


def main():
  fname = os.path.join(DATA_DIR, CSV_FNAME)
  if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

  # run this at first to make sure everything looks good
  if RUN_CHECKS:
    print 'Running checks'
    checks()
    sys.stdout.flush()

  # then run to to generate the vectorized data from the raw dump (already done)
  if REGEN_DATA:
    print 'Re-generating data'
    sys.stdout.flush()
    data, labels, ranges = subsample_and_vectorize_data(fname, LABEL, PRETTY_PRINT_LABEL)
    np.save(open(VECTOR_DATA_PATH, 'w'), data)
    data = None
    np.save(open(VECTOR_LABELS_PATH, 'w'), labels)
    labels = None
    json.dump(ranges, open(VALUE_RANGES_PATH, 'w'), indent=4)
    ranges = None

  # finally train the model
  if TRAIN == 'deep_model':
    print 'Training deep model'
    sys.stdout.flush()
    train_deep_model()
    test_deep_model()

  # you can also train a logreg
  if TRAIN == 'logistic_regression':
    print 'Training logistic regression'
    sys.stdout.flush()
    train_logistic_regression()

  # or train a random forest
  if TRAIN == 'random_forest':
    print 'Training random forest'
    sys.stdout.flush()
    top_features = train_random_forest()
    f = open(IMPORTANCES_CSV, 'w')
    f.write('feature,importance\n')
    for feature, importance in top_features:
      f.write('%s,%.4f\n' % (feature, importance))
    f.close()


def checks():
  """Run basic sanity checks on the data.
  """
  fname = os.path.join(DATA_DIR, CSV_FNAME)
  print 'Checking session uniqueness'
  check_session_uniqueness(fname)
  print 'Computing variable histogram'
  sys.stdout.flush()
  compute_variable_histograms(fname, LABEL, PRETTY_PRINT_LABEL)


def prepare_data():
  """Preprocess the data and split it into training/validation sets.

  Returns:
    processed, split data as a tuple (x_train, y_train), (x_val, y_val)
  """
  print 'Loading VECTOR_DATA_PATH'
  sys.stdout.flush()
  data = np.load(open(VECTOR_DATA_PATH))
  print 'Loading VECTOR_LABELS_PATH'
  sys.stdout.flush()
  labels = np.load(open(VECTOR_LABELS_PATH))
  print 'Loading done'
  sys.stdout.flush()

  # shuffle the data
  np.random.seed(1234)
  indices = np.arange(len(labels))
  np.random.shuffle(indices)
  data = data[indices]
  labels = labels[indices]

  # split into training and validation sets
  validation_split = 0.2
  val_samples = int(len(labels) * validation_split)
  x_train = data[:-val_samples]
  y_train = labels[:-val_samples]
  x_val = data[-val_samples:]
  y_val = labels[-val_samples:]

  # preprocess data
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_val = scaler.transform(x_val)

  print 'x_train.shape:', x_train.shape
  print 'x_val.shape:', x_val.shape
  print 'x_val average:', x_val.mean()
  print 'y_val average:', y_val.mean()
  return (x_train, y_train), (x_val, y_val)


def train_logistic_regression():
  """Train a logistic regression model on the saved data.

    baseline: 0.792
    logistic regression (no regularization): 0.871
  """
  (x_train, y_train), (x_val, y_val) = prepare_data()

  model = Sequential()
  model.add(Dense(1, input_dim=x_train.shape[1]))
  model.add(Activation('sigmoid'))
  model.compile(optimizer='adagrad',
                loss='binary_crossentropy',
                metrics=["accuracy"])
  model.fit(x_train,
            y_train,
            nb_epoch=EPOCH_COUNT,
            batch_size=32,
            validation_data=(x_val, y_val),
            verbose=2,
            shuffle=True)


def train_deep_model():
  """Train a "deep" (multiple layers of representations) model.

    baseline: 0.792
    3 layers of 128 parameters, relu, 0.2 dropout: 0.931
  """
  (x_train, y_train), (x_val, y_val) = prepare_data()

  model = Sequential()
  model.add(Dense(128, input_dim=x_train.shape[1]))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))
  model.compile(optimizer='adagrad',
                loss='binary_crossentropy',
                metrics=["accuracy"])
  model.fit(x_train,
            y_train,
            nb_epoch=EPOCH_COUNT,
            batch_size=32,
            validation_data=(x_val, y_val),
            verbose=2,
            shuffle=True)


def train_random_forest():
  """
    base settings, 200 trees:  0.934
  """
  (x_train, y_train), (x_val, y_val) = prepare_data()
  y_train = y_train.ravel()
  y_val = y_val.ravel()
  clf = RandomForestClassifier(n_estimators=FOREST_SIZE,
                               criterion='gini',
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0,
                               max_features='auto',
                               max_leaf_nodes=None,
                               bootstrap=True,
                               oob_score=False,
                               n_jobs=12,
                               random_state=None,
                               verbose=2,
                               warm_start=False,
                               class_weight=None)
  clf.fit(x_train, y_train)
  preds = clf.predict(x_val)
  acc = accuracy_score(y_val, preds)
  print 'acc:', acc
  print clf.feature_importances_

  f = open('features_names.json')
  features_names = json.load(f)

  print 'feature importances:'
  assert len(features_names) == len(clf.feature_importances_)
  unique_importances = {}
  for i in range(len(features_names)):
    name = features_names[i]
    importance = clf.feature_importances_[i]
    print name, importance
    if name not in unique_importances:
      unique_importances[name] = importance
    else:
      unique_importances[name] += importance

  print 'per unique feature:'
  tups = unique_importances.items()
  tups.sort(key=lambda x: x[1], reverse=True)
  for name, importance in tups:
    print name, importance
  return tups


def test_deep_model():
  print 'Testing deep model'
  scaler = StandardScaler()
  print 'Loading data set'
  original_data = np.load(open(VECTOR_DATA_PATH))
  scaler.fit(original_data)
  model_json = open(MODEL_PATH + '.json', 'r').read()
  model = model_from_json(model_json)
  model.load_weights(MODEL_PATH + '.h5')
  model.compile(optimizer='adagrad',
                loss='binary_crossentropy')
  histograms, ranges, size = load_value_ranges(VALUE_RANGES_PATH)
  results = {}

  #delete any existing metrics files
  for metric in histograms:
    if os.path.isfile(DATA_CSV + metric + '.csv'):
      os.remove(DATA_CSV + metric + '.csv')
  for metric in ranges:
    if os.path.isfile(DATA_CSV + metric + '.csv'):
      os.remove(DATA_CSV + metric + '.csv')

  count = 0
  iterations = 100
  print 'Testing {0:d} random data variations'.format(iterations)
  raw_results = {};
  for sample in range(0, iterations):
    count += 1
    print 'Testing {0:d}/{1:d}'.format(count, iterations)
  #for random in original_data:
    random_index = random.randint(0, len(original_data) - 1)
    random_data = original_data[random_index]
    #random_data = randomize_data(histograms, ranges, size)
#    for metric in histograms:
#      data, values = build_histogram(random_data, histograms[metric]['first'], histograms[metric]['last'])
#      data = scaler.transform(data)
#      prob = model.predict_proba(data, verbose=0)
#      stddev = np.std(prob)
#      if metric not in results:
#        results[metric] = np.array([stddev])
#      else:
#        results[metric] = np.append(results[metric], stddev)

    for metric in ranges:
      data, values = build_value(random_data, ranges[metric]['index'], ranges[metric]['min'], ranges[metric]['max'])
      if metric not in raw_results:
        raw_results[metric] = []
        for value in values:
          raw_results[metric].append([value])
      data = scaler.transform(data)
      prob = model.predict_proba(data, verbose=0)
      stddev = np.std(prob)
      if metric not in results:
        results[metric] = np.array([stddev])
      else:
        results[metric] = np.append(results[metric], stddev)
      # only keep track of records that showed a change over the sample set
      if len(prob) == len(values) and (prob.max() - prob.min()) > 0.05:
      #if len(prob) == len(values):
        for i in range(len(prob)):
          raw_results[metric][i].extend(prob[i])

  for metric in ranges:
    if metric in raw_results:
      with open(DATA_CSV + metric + '.csv', 'a') as csv_file:
        for i in range(len(raw_results[metric])):
          for y in range(len(raw_results[metric][i])):
            csv_file.write('{0:.8f},'.format(float(raw_results[metric][i][y])))
          csv_file.write('\n');

  aggregate = {}
  csv = ''
  for metric in results:
    aggregate[metric] = np.average(results[metric])
    csv += '{0},{1:.8f}\n'.format(metric, aggregate[metric])

  with open(IMPORTANCES_CSV, 'w') as csv_file:
    csv_file.write(csv)

  print '\rDone                                  '


def build_histogram(src, first, last):
  count = last - first + 1
  data = np.array([[0.0] * len(src)] * count)
  values = []
  for row in range(0, count):
    data[row] = src
    for index in range(first, last + 1):
      data[row][index] = 0.0
      values.append(int(index - first))
    data[row][first + row] = 1.0
  return data, values


def build_value(src, index, minVal, maxVal):
  minVal = int(minVal)
  maxVal = int(maxVal)
  count = min(maxVal - minVal + 1, 200)
  step = float(maxVal - minVal + 1) / float(count)
  data = np.array([[0.0] * len(src)] * count)
  values = []
  for row in range(0, count):
    data[row] = src
    value = max(minVal, min(maxVal, minVal + int(row * step)))
    data[row][index] = value
    values.append(value)
  return data, values


def randomize_data(histograms, ranges, size):
  values = np.array([0.0] * size)
  # set one random flag in each of the histograms
  for metric in histograms:
    for index in range(histograms[metric]['first'], histograms[metric]['last'] + 1):
      values[index] = 0.0
    set = random.randint(histograms[metric]['first'], histograms[metric]['last'])
    values[set] = 1.0
  # set the numeric range values randomly
  for metric in ranges:
    values[ranges[metric]['index']] = float(random.randint(int(ranges[metric]['min']), int(ranges[metric]['max'])))
  return values


def load_value_ranges(path):
  with open(path) as data_file:
    data = json.load(data_file)
  #split out the histogram values from the numeric values
  size = len(data)
  histograms = {}
  ranges = {}
  for index, entry in enumerate(data):
    metric = entry['metric']
    if 'value' in entry:
      if metric not in histograms:
        histograms[metric] = {'first': index, 'last': index}
      else:
        histograms[metric]['first'] = min(histograms[metric]['first'], index)
        histograms[metric]['last'] = max(histograms[metric]['last'], index)
    elif 'min' in entry and 'max' in entry:
      ranges[metric] = {'index': index, 'min': entry['min'], 'max': entry['max']}
  return histograms, ranges, size


if __name__ == '__main__':
  main()
