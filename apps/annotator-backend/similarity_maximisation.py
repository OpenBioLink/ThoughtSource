from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_with_jaccard(sentences, lengths):
  similarity_matrix = [[1, 0], [0, 1]]

  return calculate_with_similarity_matrix(similarity_matrix, sentences, lengths)

def calculate_with_tfidf(sentences, lengths):
  tfidf = TfidfVectorizer().fit_transform(sentences)
  # no need to normalise, since Vectorizer will return normalised tf-idf
  pairwise_similarity = tfidf * tfidf.T
  similarity_matrix = pairwise_similarity.toarray()

  return calculate_with_similarity_matrix(similarity_matrix, sentences, lengths)

# Returns elements that contain the sentence indices (one or none for each block), and the similarity score.
def calculate_with_similarity_matrix(similarity_matrix, sentences, lengths):
  sentence_elements = _create_sentence_elements(similarity_matrix, sentences, lengths)
  return _determine_top_similarities(sentence_elements, len(lengths))

# Creates sentence elements, containing similarity information to each sentence of other blocks
def _create_sentence_elements(similarity_matrix, sentences, lengths):
  # iterate over each sentence
  sentence_elements = []
  for i, sentence in enumerate(sentences):
      block_index = _get_block_index(i, lengths)

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
              similarity = similarity_matrix[i][other_sentence_index]
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

# Using similarity information of sentence elements, determines the most similar sentences.
def _determine_top_similarities(sentence_elements, texts_count):
  # Wthin all sentence elements, sort similarities to sentences of other blocks,
  # moving the most similar sentences to the front of the array
  _sort_similarities_for_all(sentence_elements)

  # Sort sentence elements themselves,
  # moving those with highest similarity to sentences of other blocks to the fron
  sentence_elements.sort(key=lambda d: _sum_block_similarities(d), reverse=True)

  # UNCOMMENT to check/verify similarity comparison and id removal
  #print("Initial sorted sentence_elements")
  #print(sentence_elements)

  # Extract indices and similarity score of the most similar sentences,
  # then remove those sentences and repeat until no sentences are left
  top_similarities = []
  while len(sentence_elements) > 0:
    top_candidate = sentence_elements[0]

    # get all sentence indices relevant to top candidate
    indices = [top_candidate['index']]
    for similarity_elements in top_candidate['block_similarities']:
      if len(similarity_elements) is 0:
        continue
      indices.append(similarity_elements[0]['other_index'])

    # store top similarity
    similarity_score = _sum_block_similarities(top_candidate) / texts_count;

    if similarity_score > 0:
      top_similarity = {
          'similarity_score': similarity_score,
          'indices': indices
      }
      top_similarities.append(top_similarity)
    else:
      break

    # remove ids from top candidate everywhere
    for remove_id in indices:
      sentence_elements = _remove_index_everywhere(sentence_elements, remove_id)
    
    # sort by summing similarities again, they could have changed
    sentence_elements.sort(key=lambda d: _sum_block_similarities(d), reverse=True)

    # UNCOMMENT to check/verify similarity comparison and id removal
    #print(f"similarity score {similarity_score}")
    #print(f"removing {indices}")
    #print("sentence_elements after iteration")
    #print(sentence_elements)
    
  return top_similarities

# Each sentence element stores the similarity to each sentence of each (other) block.
# Reorders these similarity elements so that the most similar sentence indices of each other block
# are at the array's front.
def _sort_similarities_of_sentence_element(sentence_element):
  blocks = sentence_element['block_similarities']
  for i, block in enumerate(blocks):
    block.sort(key=lambda d: d['similarity'], reverse=True)

# Moves the top similarities to front of the similarity arrays within all sentence elements
def _sort_similarities_for_all(sentence_elements):
  for sentence_element in sentence_elements:
    _sort_similarities_of_sentence_element(sentence_element)

# Sums the similarity of a sentence element, using the first (if any) similarity entry of each block
def _sum_block_similarities(sentence_element):
  similarity_sum = 0
  for similarity_elements in sentence_element['block_similarities']:
    if len(similarity_elements) is 0:
      continue
    similarity_sum += similarity_elements[0]['similarity']
  return similarity_sum

# Each sentence should only be highlighted once
# -> after a sentence has been chosen when maximising similarities,
# remove it from data structures so that later iterations do not consider it anymore
def _remove_index_everywhere(sentence_elements, index_to_remove):
  sentence_elements = [item for item in sentence_elements if item['index'] is not index_to_remove]
  # Iterate over each sentence
  for sentence_element in sentence_elements:
    # for each other block, remove all similarity_elements with index_to_remove
    for block_id, block in enumerate(sentence_element['block_similarities']):
      sentence_element['block_similarities'][block_id] = [item for item in sentence_element['block_similarities'][block_id] if item['other_index'] is not index_to_remove]
  return sentence_elements

# Determines inDex of corresponding block for the index of a sentence (offset 0 for both)
# Example:
# data = [['sentence 0', 'sentence 1', 'sentence 2'], ['sentence 3', 'sentence 4']]
# lengths = [3, 2]
# _get_block_index(2, lengths) will return 0
# _get_block_index(3, lengths) will return 1
def _get_block_index(index, lengths):
    block_index = 0
    current_length = 0
    while block_index + 1 < len(lengths) and index >= lengths[block_index] + current_length:
        current_length += lengths[block_index]
        block_index += 1
    return block_index

# Unused method - to verify functionality of _get_block_index
def _test_it(param):
    print(_get_block_index(0, param))
    print(_get_block_index(3, param))
    print(_get_block_index(4, param))
    print(_get_block_index(5, param))
    print(_get_block_index(6, param))

# Unused method - in case we wanted to use offsets of blocks, rather than sentence array lengths
def _get_block_index_with_offsets(index, offsets):
    block_index = 0
    while block_index + 1 < len(offsets) and index >= offsets[block_index + 1]:
        block_index += 1
    return block_index

# Unused method - splits input text by full stops, whitespaces trimmed. No empty strings.
def _split_text_into_segments(input):
  segments = (segment.strip() for segment in input.rsplit(". ") if segment.strip())
  segments = list(segment + ("." if not segment.endswith(".") else "") for segment in segments)
  return segments

# Unused method - logic to obtain color for a sentence, dependent on it's similarity rank
def _get_color_for_sentence(sentence_index, similarities):
  colors_bright = ['#FBFA30', '#3CFA72', '#32B5FF']
  colors = ['#20729E', '#249945', '#7d2cc7']
  for color_index, color in enumerate(colors):
    if color_index >= len(similarities):
      return None
    if sentence_index in similarities[color_index]['indices']:
      return color
  return None
