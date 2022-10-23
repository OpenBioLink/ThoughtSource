import React, { FC, useEffect, useState } from 'react';
import CotData, { findExistingAnnotation } from '../../dtos/CotData';
import CotOutputElement from '../cotoutputelement/CotOutputElement';
import { FAVORED, SentenceElement, SentenceElementDict, SimilarityInfo } from '../datasetentry/DatasetEntry';
import styles from './DatasetEntryElement.module.scss';

interface DatasetEntryElementProps {
  cotData: CotData
  username: string
  anyUpdatePerformed: () => void
}

const DatasetEntryElement: FC<DatasetEntryElementProps> = (props) => {
  const [similarities, setSimilarities] = useState<SimilarityInfo[]>([])
  const [bestCotIndex, setBestCotIndex] = useState<number>(props.cotData.generated_cot.findIndex(cotData => cotData['isFavored']))


  // Previously in constructor
  const answers = props.cotData.generated_cot.map(cotData => cotData.cot)
  const splitAnswers = answers.map(answer => answer.split(". ")
    .map((sentence, index, arr) => (index != arr.length - 1) ? sentence + ". " : sentence))

  const lengths = splitAnswers.map(answerArray => answerArray.length)
  const sentences = splitAnswers.flatMap(answerArray => answerArray)

  // Previously in componentDidMount
  useEffect(() => {
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sentences: sentences,
        lengths: lengths
      })
    }

    fetch('http://localhost:5000/textcompare', requestOptions)
      .then(response => response.json())
      .then((data: SimilarityInfo[]) => {
        data.forEach((value, index) => value.index = index)
        setSimilarities(data)
      })
      .catch(error => {
        console.log("Error fetching similarities")
        console.log(error)
      })
  }, [])  // Pass an empty array to run callback on mount only.

  // Previously methods of class
  function updateBestCot(bestCotIndex: number) {
    setBestCotIndex(bestCotIndex)

    props.cotData.generated_cot?.forEach((cotOutput, index) => {
      cotOutput.isFavored = bestCotIndex == index
      // find annotation for it
      const annotation = findExistingAnnotation(cotOutput, FAVORED)
    })

    props.anyUpdatePerformed()
  }

  function getBlockIndex(index: number) {
    let blockIndex = 0
    let currentLength = 0
    while (blockIndex + 1 < lengths.length && index >= lengths[blockIndex] + currentLength) {
      currentLength += lengths[blockIndex]
      blockIndex++
    }
    return blockIndex
  }

  // Previously in render method
  const sentenceElementsDict = props.cotData.generated_cot.reduce((aggr, value, index) => {
    aggr[index] = []
    return aggr
  }, {} as SentenceElementDict)

  sentences.forEach((sentence, index) => {
    const blockIndex = getBlockIndex(index)

    // by contract, sentence can only appear once in entire similarities information
    const similarity = similarities?.find(similarity => similarity.indices?.find(i => i == index) != null)

    const sentenceElement = { sentence: sentence, similarityIndex: similarity?.index, similarityScore: similarity?.similarity_score } as SentenceElement
    sentenceElementsDict[blockIndex].push(sentenceElement)
  })

  const resultElements = props.cotData.generated_cot?.map((cotOutput, index) =>
    <li key={index}>
      <CotOutputElement
        cotOutput={cotOutput}
        sentenceElements={sentenceElementsDict[index]}
        bestCot={bestCotIndex == index}
        correctAnswer={props.cotData.answer[0]}
        username={props.username}
        updateBestCot={() => updateBestCot(index)}
        updateExportFile={props.anyUpdatePerformed} />
    </li>)

  return <div className={styles.DatasetEntryElement}>
    <div className={styles.EntryHeader}>
      <h3>Question</h3>
      <span>{props.cotData.question}</span>
      <br />
      <span style={{ fontStyle: "oblique" }}>Correct answer: {props.cotData.answer}</span>
    </div>
    <ul className={styles.OutputsContainer}>
      {resultElements}
    </ul>
  </div>
}

export default DatasetEntryElement;
