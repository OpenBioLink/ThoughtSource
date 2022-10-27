import os
from datetime import timedelta
from operator import le

from flask import Flask, jsonify, make_response, request, session

from cors_handling import cors_handling
from similarity_maximisation import (calculate_with_jaccard,
                                     calculate_with_tfidf)

app = Flask(__name__)

app.secret_key = b'\x1d\x12\xc72\xf2\xd9\xcd\x92\x87/\x87P\x8e\xfe\xa0\xff[F\xe5S/\xa1\\\xe9'
SESSION_ID = 'SESSION_ID'
sessions_dict = {}

@app.before_request
def session_handling():
    if SESSION_ID not in session:
      # Generate new session id
      new_sid = os.urandom(10)
      while new_sid in sessions_dict:
        new_sid = os.urandom(10)
      session[SESSION_ID] = new_sid
      sessions_dict[new_sid] = session
      print(f"New session {new_sid}")
    else:
      print(f"Existing session {session[SESSION_ID]}")

    # Make session permanent, default lifetime is 31 days
    session.permanent = True
    #app.permanent_session_lifetime = timedelta(minutes=5)

@app.route("/checkin", methods=['GET', 'POST', 'OPTIONS'])
@cors_handling
def checkin():
  # TODO this
  # Keep sessions for each username as well
  # Think of fitting datastructure to prioritise either session or username (probably session first, then username)
  return {'text': "OK"}

@app.route("/backup", methods=['POST', 'OPTIONS'])
@cors_handling
def backup():
    # TODO
    return "TODO"

@app.route("/compareall", methods=['POST', 'OPTIONS'])
@cors_handling
def compare_all():
    data = request.get_json()
    username = data['username']
    entries = data['entries']

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
  #similarities_by_methods['jaccard'] = calculate_with_jaccard(sentences, lengths)
  return similarities_by_methods

@app.route("/textcompare", methods=['POST', 'OPTIONS'])
@cors_handling
def textcompare():
    data = request.get_json()
    sentences = data['sentences']
    lengths = data['lengths']
    username = data['username']

    similarities_by_methods = {}
    similarities_by_methods['tfidf'] = calculate_with_tfidf(sentences, lengths)
    #similarities_by_methods['jaccard'] = calculate_with_jaccard(sentences, lengths)
    return jsonify(similarities_by_methods)

if __name__ == "__main__":
    app.run(debug = True)
