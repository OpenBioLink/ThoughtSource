import atexit
import json
import os
import pickle
from asyncore import write
from datetime import timedelta
from operator import le

from flask import Flask, jsonify, make_response, request, session

from cors_handling import cors_handling
from similarity_maximisation import (calculate_with_jaccard,
                                     calculate_with_tfidf)

app = Flask(__name__)

app.secret_key = b'\x1d\x12\xc72\xf2\xd9\xcd\x92\x87/\x87P\x8e\xfe\xa0\xff[F\xe5S/\xa1\\\xe9'
SESSION_ID_KEY = 'session_id'
USERNAME_KEY = 'username'
FILE_NAME_KEY = 'filename'
FILE_CONTENT_KEY = 'filecontent'

DICT_FILE_NAME = 'sessions_dict.pickle'
sessions_dict = {}

@app.before_request
def session_handling():
  handle_session_id()
  handle_session_entry()

  # Make session permanent, default lifetime is 31 days
  session.permanent = True
  #app.permanent_session_lifetime = timedelta(minutes=5)

def handle_session_id():
  if SESSION_ID_KEY not in session:
    # Generate new session id
    new_sid = os.urandom(10)
    while new_sid in sessions_dict:
      new_sid = os.urandom(10)
    session[SESSION_ID_KEY] = new_sid
    print(f"New session {new_sid}")
  else:
    print(f"Existing session {session[SESSION_ID_KEY]}")

def handle_session_entry():
  sid = session[SESSION_ID_KEY]
  if sid not in sessions_dict:
    sessions_dict[sid] = {}

def get_session_entry():
  if SESSION_ID_KEY not in session:
    print(f"No session ID when retrieving session data")
    return
  
  sid = session[SESSION_ID_KEY]
  if sid not in sessions_dict:
    print(f"No dict entry for session ID")
    return

  return sessions_dict[sid]

@app.route("/checkin", methods=['GET', 'POST', 'OPTIONS'])
@cors_handling
def checkin():
  session_entry = get_session_entry()
  if not session_entry:
    return {}

  username = session_entry.get(USERNAME_KEY)
  file_name = session_entry.get(FILE_NAME_KEY)
  file_content = session_entry.get(FILE_CONTENT_KEY)

  return {
    USERNAME_KEY: username,
    FILE_NAME_KEY: file_name,
    FILE_CONTENT_KEY: file_content
    }

@app.route("/backup", methods=['POST', 'OPTIONS'])
@cors_handling
def backup():
  session_entry = get_session_entry()
  data = request.get_json()

  session_entry[USERNAME_KEY] = data[USERNAME_KEY]
  session_entry[FILE_NAME_KEY] = data[FILE_NAME_KEY]
  session_entry[FILE_CONTENT_KEY] = data[FILE_CONTENT_KEY]

  return {'text': "Backup OK"}

@app.route("/compareall", methods=['POST', 'OPTIONS'])
@cors_handling
def compare_all():
    entries = request.get_json()

    similarities_for_cots = []
    for entry in entries:
      sentences = entry['sentences']
      lengths = entry['lengths']
      similarities_by_methods = similarities_for_multple_methods(sentences, lengths)
      similarities_for_cots.append(similarities_by_methods)
    return jsonify(similarities_for_cots)

def similarities_for_multple_methods(sentences, lengths):
  similarities_by_methods = {}
  similarities_by_methods['tfidf'] = calculate_with_tfidf(sentences, lengths)
  similarities_by_methods['jaccard'] = calculate_with_jaccard(sentences, lengths)
  return similarities_by_methods

@app.route("/textcompare", methods=['POST', 'OPTIONS'])
@cors_handling
def textcompare():
    data = request.get_json()
    sentences = data['sentences']
    lengths = data['lengths']

    similarities_by_methods = {}
    similarities_by_methods['tfidf'] = calculate_with_tfidf(sentences, lengths)
    similarities_by_methods['jaccard'] = calculate_with_jaccard(sentences, lengths)
    return jsonify(similarities_by_methods)

def read_sessions_dict_from_file():
  print("Load sessions dict from file")
  try:
    with open(DICT_FILE_NAME, 'rb') as file:
      loaded_dict = pickle.load(file)
      sessions_dict.update(loaded_dict)

  except FileNotFoundError as error:
    print("No dict file yet")
  except Exception as anything:
    print(anything)

# Careful - Debug mode calls this twice, overwrites with empty file the second time around
def write_sessions_dict_to_file():
  try:
    # write to file binary
    with open(DICT_FILE_NAME, 'wb') as file:
      pickle.dump(sessions_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
  except IOError as error:
    print("Error writing dict file")

read_sessions_dict_from_file()
atexit.register(write_sessions_dict_to_file)

if __name__ == "__main__":
  app.run(debug = True)
