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
import json
import keras
import numpy as np
import os
import random
import sys

from data_utils import subsample_and_vectorize_data

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

from sklearn.preprocessing import StandardScaler

REGEN_DATA = False

# Training bounce or conversion
#LABEL = 'is_converted'
#PRETTY_PRINT_LABEL = 'conversion'
LABEL = 'is_bounce'
PRETTY_PRINT_LABEL = 'bounce'

# Including bounce rate data
DATA_DIR = './data'
RESULTS_DIR = './results'
CSV_FNAME = 'beacon.csv'

TRAIN = 'deep_model'
VECTOR_DATA_PATH = DATA_DIR + '/' + PRETTY_PRINT_LABEL + '_data.npy'
VECTOR_LABELS_PATH = DATA_DIR + '/' + PRETTY_PRINT_LABEL + '_labels.npy'
MODEL_PATH = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_model'
VALUE_RANGES_PATH = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_value_ranges.json'
IMPORTANCES_CSV = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_importances.csv'
DATA_CSV = RESULTS_DIR + '/' + PRETTY_PRINT_LABEL + '_' + TRAIN + '_data_'

EPOCH_COUNT = 100

# Features to include as a starting point before iterating on all other available features
starting_features = []

#Features to plot the probability distribution for.  If this is set then only these features will
# be evaluated (and it probably only makes sense if starting_features is empty).
#test_features = ['median_timers_domready','session_avg_loadtime','median_timers_render']
test_features = []


def main():
  if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

  # then run to to generate the vectorized data from the raw dump (already done)
  if REGEN_DATA or \
      not os.path.exists(VECTOR_DATA_PATH) or \
      not os.path.exists(VECTOR_LABELS_PATH) or \
      not os.path.exists(VALUE_RANGES_PATH):
    print 'Re-generating data'
    csv_fname = os.path.join(DATA_DIR, CSV_FNAME)
    sys.stdout.flush()
    data, labels, ranges = subsample_and_vectorize_data(csv_fname, LABEL, PRETTY_PRINT_LABEL)
    with open(VECTOR_DATA_PATH, 'wb') as file:
      np.save(file, data)
    data = None
    with open(VECTOR_LABELS_PATH, 'wb') as file:
      np.save(file, labels)
    labels = None
    with open(VALUE_RANGES_PATH, 'wb') as file:
      json.dump(ranges, file, indent=4)
    ranges = None

  features = load_feature_names()
  (x_train_full, y_train), (x_val_full, y_val) = prepare_data()
  train_rows, train_cols = x_train_full.shape
  val_rows, val_cols = x_val_full.shape

  # Figure out how many columns we need for the known starting features
  fname = os.path.join(RESULTS_DIR, PRETTY_PRINT_LABEL + '_accuracy_test')
  base_columns = 0;
  for name in starting_features:
    if name in features:
      fname += "." + name
      base_columns += features[name]['end'] - features[name]['start'] + 1
  fname += ".csv"

  # Try training each feature against the data set individually
  feature_count = len(features)
  feature_num = 0
  with open(fname, 'wb', 1) as out:
    for name, feature in features.iteritems():
      if not len(test_features) or name in test_features:
        feature_num += 1

        #Build an input data set with just the columns we care about
        count = feature['end'] - feature['start'] + 1
        x_train = np.zeros((train_rows, base_columns + count))
        x_val = np.zeros((val_rows, base_columns + count))
        col = 0
        # Populate the starting features
        for n in starting_features:
          if n == name:
            continue
          if n in features:
            for column in xrange(features[n]['start'], features[n]['end'] + 1):
              x_train[:, col] = x_train_full[:, column]
              x_val[:, col] = x_val_full[:, column]
              col += 1
        # Populate the features we are testing
        for column in xrange(feature['start'], feature['end'] + 1):
          x_train[:,col] = x_train_full[:,column]
          x_val[:, col] = x_val_full[:, column]
          col += 1

        # normalize the data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

        # Run the actual training
        print '[{0:d}/{1:d}] Training deep model on {2} ({3:d} columns)'.format(feature_num, feature_count, name, col)
        sys.stdout.flush()
        acc, model = train_deep_model(x_train, y_train, x_val, y_val)
        print '{0} Accuracy: {1:0.4f}'.format(name, acc)
        sys.stdout.flush()
        out.write('{0},{1:0.4f}\n'.format(name,acc))

        # Test the varous values for the feature
        if len(test_features):
          max_val = 100000
          min_val = 100
          step_size = 100
          count = (max_val - min_val) / step_size
          original_values = np.array([[0.0]] * count)
          row = 0
          for value in xrange(100, 100000, 100):
            original_values[row] = value
            row += 1
          data = scaler.transform(original_values)
          prob = model.predict_proba(data, verbose=0)
          with open(os.path.join(RESULTS_DIR, PRETTY_PRINT_LABEL + '_values_' + name), 'wb') as v:
            for row in xrange(0, count):
              value = original_values[row][0]
              probability = prob[row][0]
              v.write('{0:d},{1:f}\n'.format(int(value), probability))


class StoppedImproving(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.accuracy_history = []
    self.max_val = 0.0

  def on_epoch_end(self, epoch, logs={}):
    current_val = logs.get('val_acc')
    self.max_val = max(self.max_val, current_val)
    self.accuracy_history.append(current_val)
    count = len(self.accuracy_history)
    if count > 2:
      changed = False
      baseline = round(self.accuracy_history[0], 3)
      for index in xrange(1, count):
        val = round(self.accuracy_history[index], 3)
        if baseline != val:
          changed = True
          break
      if changed:
        all_below = True
        for index in xrange(0, count):
          val = self.accuracy_history[index]
          if val >= self.max_val:
            all_below = False
      if not changed or all_below:
        print "Stopped improving, stopping training"
        self.model.stop_training = True
      self.accuracy_history.pop(0)


def load_feature_names():
  # Load the list of known features and reduce it to unique names and the columns they occupy
  with open('features_names.json', 'rb') as f:
    raw_names = json.load(f)
  features = {}
  for index, name in enumerate(raw_names):
    if name in features:
      entry = features[name]
      entry['start'] = min(entry['start'], index)
      entry['end'] = max(entry['end'], index)
    else:
      entry = {'start': index, 'end': index}
    features[name] = entry

  return features


def prepare_data():
  """Preprocess the data and split it into training/validation sets.

  Returns:
    processed, split data as a tuple (x_train, y_train), (x_val, y_val)
  """
  print 'Loading VECTOR_DATA_PATH'
  sys.stdout.flush()
  with open(VECTOR_DATA_PATH, 'rb') as file:
    data = np.load(file)
  print 'Loading VECTOR_LABELS_PATH'
  sys.stdout.flush()
  with open(VECTOR_LABELS_PATH, 'rb') as file:
    labels = np.load(file)
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

  print 'x_train.shape:', x_train.shape
  print 'x_val.shape:', x_val.shape
  print 'x_val average:', x_val.mean()
  print 'y_val average:', y_val.mean()
  return (x_train, y_train), (x_val, y_val)


def train_deep_model(x_train, y_train, x_val, y_val):
  """Train a "deep" (multiple layers of representations) model.

    baseline: 0.792
    3 layers of 128 parameters, relu, 0.2 dropout: 0.931
  """
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
  cb = StoppedImproving()
  model.fit(x_train,
            y_train,
            nb_epoch=EPOCH_COUNT,
            batch_size=32,
            validation_data=(x_val, y_val),
            verbose=2,
            shuffle=True,
            callbacks=[cb])
  return cb.max_val, model


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
