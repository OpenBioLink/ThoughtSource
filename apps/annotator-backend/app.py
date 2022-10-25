import os
from datetime import timedelta

from flask import Flask, jsonify, make_response, request, session
from sklearn.feature_extraction.text import TfidfVectorizer

from cors_handling import cors_handling

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
  return {'text': "OK"}

@app.route("/backup", methods=['POST', 'OPTIONS'])
@cors_handling
def backup():
    # TODO
    return "TODO"

@app.route("/textcompare", methods=['POST', 'OPTIONS'])
@cors_handling
def textcompare():
    data = request.get_json()
    sentences = data['sentences']
    lengths = data['lengths']
    username = data['username']

    sentence_elements = create_sentence_elements(sentences, lengths)
    top_similarities = determine_top_similarities(sentence_elements, len(lengths))
    return jsonify(top_similarities)

# returns index of block for index of sentence (offset 0 for both)
def get_block_index(index, lengths):
    block_index = 0
    current_length = 0
    while block_index + 1 < len(lengths) and index >= lengths[block_index] + current_length:
        current_length += lengths[block_index]
        block_index += 1
    return block_index

def get_block_index_with_offsets(index, offsets):
    block_index = 0
    while block_index + 1 < len(offsets) and index >= offsets[block_index + 1]:
        block_index += 1
    return block_index

def test_it(param):
    print(get_block_index(0, param))
    print(get_block_index(3, param))
    print(get_block_index(4, param))
    print(get_block_index(5, param))
    print(get_block_index(6, param))

def sort_similarity_elements(sentence_element):
  blocks = sentence_element['block_similarities']
  for i, block in enumerate(blocks):
    block.sort(key=lambda d: d['similarity'], reverse=True)

def sort_all_similarity_elements(sentence_elements):
  for similarity_elements in sentence_elements:
    sort_similarity_elements(similarity_elements)

def sum_block_similarities(sentence_element):
  similarity_sum = 0
  for similarity_elements in sentence_element['block_similarities']:
    if len(similarity_elements) is 0:
      continue
    similarity_sum += similarity_elements[0]['similarity']
  return similarity_sum

def remove_index_everywhere(sentence_elements, index_to_remove):
  sentence_elements = [item for item in sentence_elements if item['index'] is not index_to_remove]
  # Iterate over each sentence
  for sentence_element in sentence_elements:
    # for each other block, remove all similarity_elements with index_to_remove
    for block_id, block in enumerate(sentence_element['block_similarities']):
      sentence_element['block_similarities'][block_id] = [item for item in sentence_element['block_similarities'][block_id] if item['other_index'] is not index_to_remove]
  return sentence_elements

def create_sentence_elements(sentences, lengths):
  tfidf = TfidfVectorizer().fit_transform(sentences)
  # no need to normalise, since Vectorizer will return normalised tf-idf
  pairwise_similarity = tfidf * tfidf.T
  result_array = pairwise_similarity.toarray()
  
  # PRINT to show similarity matrix
  #print(result_array)

  # iterate over each sentence
  sentence_elements = []
  for i, sentence in enumerate(sentences):
      block_index = get_block_index(i, lengths)

      sentence_element = {
          'index': i,
          'block_similarities': []
      }

      # Iterate over blocks
      current_length = 0
      for j, length in enumerate(lengths):
          if j == block_index:
              # Skip self
              current_length += length
              continue
          # Iterate over all sentences of other blocks
          similarity_elements = []
          for other_sentence_index in range(current_length, current_length + length):
              similarity = result_array[i][other_sentence_index]
              if similarity > 0:
                  similarity_elements.append({
                      'other_index': other_sentence_index,
                      'similarity': similarity
                  })
          current_length += length
      # Push array of similarity_elements for each block
          sentence_element['block_similarities'].append(similarity_elements)
      sentence_elements.append(sentence_element)
  return sentence_elements

def determine_top_similarities(sentence_elements, texts_count):
  # PRINT these to verify functionality... it worked.
  sort_all_similarity_elements(sentence_elements)
  sentence_elements.sort(key=lambda d: sum_block_similarities(d), reverse=True)

  # UNCOMMENT to check/verify similarity comparison and id removal
  #print("Initial sorted sentence_elements")
  #print(sentence_elements)

  top_similarities = []
  highest_similarity = sum_block_similarities(sentence_elements[0])
  while len(sentence_elements) > 0:
    top_candidate = sentence_elements[0]

    # get all ids relevant to top candidate
    indices = [top_candidate['index']]
    for similarity_elements in top_candidate['block_similarities']:
      if len(similarity_elements) is 0:
        continue
      indices.append(similarity_elements[0]['other_index'])

    # store top similarity
    similarity_score = sum_block_similarities(top_candidate) / texts_count;
    #print(f"similarity score {similarity_score}")

    if similarity_score > 0:
      top_similarity = {
          'similarity_score': similarity_score,
          'indices': indices
      }
      top_similarities.append(top_similarity)

    # remove ids from top candidate everywhere (TODO: store them)
    #print(f"removing {indices}")
    for remove_id in indices:
      sentence_elements = remove_index_everywhere(sentence_elements, remove_id)
    
    # sort by summing similarities again, they could have changed
    sentence_elements.sort(key=lambda d: sum_block_similarities(d), reverse=True)

    # UNCOMMENT to check/verify similarity comparison and id removal
    #print("sentence_elements after iteration")
    #print(sentence_elements)
    
  return top_similarities

# Returns all segments, whitespaces trimmed. No empty strings.
def split_text_into_segments(input):
  segments = (segment.strip() for segment in input.rsplit(". ") if segment.strip())
  segments = list(segment + ("." if not segment.endswith(".") else "") for segment in segments)
  return segments

colors_bright = ['#FBFA30', '#3CFA72', '#32B5FF']
colors = ['#20729E', '#249945', '#7d2cc7']
def get_color_for_sentence(sentence_index, similarities):
  for color_index, color in enumerate(colors):
    if color_index >= len(similarities):
      return None
    if sentence_index in similarities[color_index]['indices']:
      return color
  return None

if __name__ == "__main__":
    app.run(debug = True)
