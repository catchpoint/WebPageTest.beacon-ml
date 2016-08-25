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
import numpy as np
import json
import random
import sys

MAXVAL = 200000
MINVAL = 0
# 'C' means string feature, 'S' means skip, 'N' means numerical feature
VARIABLE_TYPES_DICT = {
    'geo_netspeed': 'C',
    'is_converted': 'S',
    'is_bounce': 'S',  # Only present for bounce dataset - comment out o/w.
    'key': 'S',
    'max_bandwidth_kbps': 'N',
    'max_params_dom_doms': 'N',
    'max_params_dom_img': 'N',
    'max_params_dom_ln': 'N',
    'max_params_dom_res': 'N',
    'max_params_dom_script': 'N',
    'max_timers_before_dns': 'N',
    'max_timers_boomerang': 'N',
    'max_timers_boomr_fb': 'N',
    'max_timers_boomr_lat': 'N',
    'max_timers_boomr_ld': 'N',
    'max_timers_boomr_to_end': 'N',
    'max_timers_custom0': 'N',
    'max_timers_custom1': 'N',
    'max_timers_custom2': 'N',
    'max_timers_custom3': 'N',
    'max_timers_custom4': 'N',
    'max_timers_custom5': 'N',
    'max_timers_custom6': 'N',
    'max_timers_custom7': 'N',
    'max_timers_custom8': 'N',
    'max_timers_custom9': 'N',
    'max_timers_dns': 'N',
    'max_timers_domload': 'N',
    'max_timers_domready': 'N',
    'max_timers_fb_to_boomr': 'N',
    'max_timers_loaded': 'N',
    'max_timers_navst_to_boomr': 'N',
    'max_timers_renderstart': 'N',
    'max_timers_ssl': 'S',
    'max_timers_t_configfb': 'N',
    'max_timers_t_configjs': 'N',
    'max_timers_t_domloaded': 'N',
    'max_timers_t_done': 'N',
    'max_timers_t_load': 'N',
    'max_timers_t_page': 'N',
    'max_timers_t_postrender': 'N',
    'max_timers_t_prerender': 'N',
    'max_timers_t_resp': 'N',
    'max_timers_tcp': 'N',
    'median_bandwidth_kbps': 'N',
    'median_params_bat_lvl': 'N',
    'median_params_dom_doms': 'N',
    'median_params_dom_img': 'N',
    'median_params_dom_ln': 'N',
    'median_params_dom_res': 'N',
    'median_params_dom_script': 'N',
    'median_timers_before_dns': 'N',
    'median_timers_boomerang': 'N',
    'median_timers_boomr_fb': 'N',
    'median_timers_boomr_lat': 'N',
    'median_timers_boomr_ld': 'N',
    'median_timers_boomr_to_end': 'N',
    'median_timers_custom0': 'N',
    'median_timers_custom1': 'N',
    'median_timers_custom2': 'N',
    'median_timers_custom3': 'N',
    'median_timers_custom4': 'N',
    'median_timers_custom5': 'N',
    'median_timers_custom6': 'N',
    'median_timers_custom7': 'N',
    'median_timers_custom8': 'N',
    'median_timers_custom9': 'N',
    'median_timers_dns': 'N',
    'median_timers_domload': 'N',
    'median_timers_domready': 'N',
    'median_timers_fb_to_boomr': 'N',
    'median_timers_loaded': 'N',
    'median_timers_navst_to_boomr': 'N',
    'median_timers_renderstart': 'N',
    'median_timers_ssl': 'S',
    'median_timers_t_configfb': 'N',
    'median_timers_t_configjs': 'N',
    'median_timers_t_domloaded': 'N',
    'median_timers_t_done': 'N',
    'median_timers_t_load': 'N',
    'median_timers_t_page': 'N',
    'median_timers_t_postrender': 'N',
    'median_timers_t_prerender': 'N',
    'median_timers_t_resp': 'N',
    'median_timers_tcp': 'N',
    'mobile_connection_type': 'C',
    'params_cpu_cnc': 'C',
    'params_scr_bpp': 'C',
    'params_scr_dpx': 'N',
    'params_scr_mtp': 'N',
    'params_scr_orn': 'C',
    'params_scr_xy': 'S',
    'session_end': 'S',
    'session_id': 'S',
    'session_length': 'S',
    'session_start': 'S',
    # 'session_totalloadtime': 'N',  # Name of the field in orig data
    'session_total_loadtime': 'S',  # Name of the field in bounce rate data
    'session_avg_loadtime': 'N',  # Only found in bounce rate data
    'spdy': 'S',
    'ssl': 'S',
    'timers_missing': 'C',
    'user_agent_device_type': 'C',
    'user_agent_family': 'C',
    'user_agent_isp': 'S',
    'user_agent_major': 'C',
    'user_agent_manufacturer': 'C',
    'user_agent_minor': 'S',
    'user_agent_mobile': 'S',
    'user_agent_model': 'S',
    'user_agent_os': 'C',
    'user_agent_osversion': 'S',
}

def _generate_header_dict(line):
  header_list = line.strip().split('|')
  header_to_position = {}
  for i in xrange(len(header_list)):
    header_to_position[header_list[i]] = i
  return header_to_position, header_list


def check_session_uniqueness(fname):
  with open(fname, 'rb') as f:
    header_to_position, _ = _generate_header_dict(f.readline())

    if len(header_to_position) > len(VARIABLE_TYPES_DICT):
      not_found = []
      for k in header_to_position:
        if k not in VARIABLE_TYPES_DICT:
          not_found.append(k)
      if not_found:
        assert False, ('Fields in header that are not defined in '
                       'VARIABLE_TYPES_DICT: ') + str(not_found)

    if len(VARIABLE_TYPES_DICT) > len(header_to_position):
      not_found = []
      for k in VARIABLE_TYPES_DICT:
        if k not in header_to_position:
          not_found.append(k)
      if not_found:
        assert False, ('Expected fields from VARIABLE_TYPES_DICT not found in '
                       'header: ') + str(not_found)

    key_set = set()
    sess_set = set()
    total_lines = 0
    for line in f:
      line = line.strip()
      total_lines += 1
      if total_lines % 5000 == 1:
        print 'Processing line: %d' % total_lines
        sys.stdout.flush()
      values = line[1:-1].split('"|"')
      assert len(values) == len(header_to_position)

      key = values[header_to_position['key']]
      sess_id = values[header_to_position['session_id']]
      key_set.add(key)
      sess_set.add(sess_id)

  print 'total unique sessions:', len(sess_set)
  print 'total unique keys:', len(key_set)
  print 'total events:', total_lines
  # assert len(sess_set) == total_lines


def compute_variable_histograms(fname, label, pretty_print_label, display=True):
  with open(fname, 'rb') as f:
    header_to_position, header_list = _generate_header_dict(f.readline())
    histograms = dict([(name, set()) for name in VARIABLE_TYPES_DICT])
    converted = 0
    not_converted = 0
    total_lines = 0
    for line in f:
      line = line.strip()
      values = line[1:-1].split('"|"')
      assert len(values) == len(header_list)

      render = values[header_to_position['median_timers_renderstart']].strip()
      if not len(render) or float(render) <= 0:
        continue

      total_lines += 1
      for value, name in zip(values, header_list):
        assert name in VARIABLE_TYPES_DICT, ('Header %s not found in '
                                             'VARIABLE_TYPES_DICT') % name
        vtype = VARIABLE_TYPES_DICT[name]
        if vtype == 'C':
          histograms[name].add(value)
        elif vtype == 'N':
          try:
            value = min(MAXVAL, max(MINVAL, float(value)))
          except:
            if value:
              print 'WARNING: treating feature %s with value %s as 0' % (name,
                                                                         value)
            value = 0
        elif vtype == 'S':
          pass
        else:
          raise Exception('Unknown variable type ' + vtype)

      if values[header_to_position[label]] == 't':
        converted += 1
      else:
        not_converted += 1

  #assert converted + not_converted == total_lines
  ratio = float(converted) / total_lines

  if display:
    print 'Histograms:'
    items = histograms.items()
    items.sort(key=lambda x: len(x[1]))
    for item in items:
      n = item[0]
      h = item[1]
      print n, len(h)
      sys.stdout.flush()
    print '-'
    print '%s ratio: %f' % (pretty_print_label, ratio)

  return ratio, total_lines, dict([(name, len(histograms[name])) for name in VARIABLE_TYPES_DICT])


def subsample_and_vectorize_data(fname, label, pretty_print_label):
  subsample_factor, nb_samples, histograms = compute_variable_histograms(fname, label, pretty_print_label)

  with open(fname, 'rb') as f:
    header_to_position, header_list = _generate_header_dict(f.readline())

  nb_kept_samples = int(subsample_factor * nb_samples) * 2
  print 'will keep {0:d} samples from {1:d} records.'.format(nb_kept_samples, nb_samples)

  variable_start_indices = {}
  value_indices = dict([(name, {}) for name in VARIABLE_TYPES_DICT])

  line_length = 0
  feature_names = []
  for name in header_list:
    assert name in VARIABLE_TYPES_DICT
    vtype = VARIABLE_TYPES_DICT[name]

    if vtype == 'N':
      variable_start_indices[name] = line_length
      line_length += 1
      feature_names.append(name)
    if vtype == 'C':
      variable_start_indices[name] = line_length
      line_length += histograms[name]
      feature_names += [name] * histograms[name]
  with open('features_names.json', 'wb') as f:
    json.dump(feature_names, f)

  data = np.zeros((nb_kept_samples, line_length))
  print 'data.shape:', data.shape
  labels = np.zeros((nb_kept_samples,))

  value_ranges = [None] * line_length
  with open(fname, 'rb') as f:
    header_to_position, _ = _generate_header_dict(f.readline())
    i = 0.
    for line in f:
      line = line.strip()
      values = line[1:-1].split('"|"')
      assert len(values) == len(header_to_position)

      render = values[header_to_position['median_timers_renderstart']].strip()
      if not len(render) or float(render) <= 0:
        continue

      keep = True

      if values[header_to_position[label]] == 't':
        labels[i] = 1.0
      else:
        if random.random() > subsample_factor:
          keep = False

      for value, name in zip(values, header_list):
        assert name in VARIABLE_TYPES_DICT, ('Header %s not found in '
                                             'VARIABLE_TYPES_DICT') % name
        vtype = VARIABLE_TYPES_DICT[name]
        if vtype == 'C':
          value_dict = value_indices[name]
          if value in value_dict:
            offset = value_dict[value]
          else:
            offset = len(value_dict)
            value_dict[value] = offset
            value_indices[name] = value_dict
          index = variable_start_indices[name] + offset
          if value_ranges[index] is None:
            value_ranges[index] = {'metric': name, 'value': value}
          if keep:
            data[i, index] = 1.0
        elif vtype == 'N':
          try:
            value = float(min(MAXVAL, max(MINVAL, float(value))))
          except:
            if value:
              print 'WARNING: treating feature %s with value %s as 0' % (name,
                                                                         value)
            value = 0.0
          index = variable_start_indices[name]
          if value_ranges[index] is None:
            value_ranges[index] = {'metric': name, 'min': value, 'max': value}
          if value < value_ranges[index]['min']:
            value_ranges[index]['min'] = value
          if value > value_ranges[index]['max']:
            value_ranges[index]['max'] = value
          if keep:
            data[i, index] = value
        elif vtype == 'S':
          continue
        else:
          raise Exception('Unknown variable type ' + vtype)
        if i > nb_kept_samples:
          print 'collected', nb_kept_samples, 'total samples.'
          break
      if keep:
        i += 1

  print 'y.shape:', labels.shape
  print 'y.mean:', labels.mean()
  return data, labels, value_ranges
