import { FC, useState } from 'react';
import CotData, { findExistingAnnotation, SentenceElement, SentenceElementDict, SimilarityInfo } from '../../dtos/CotData';
import CotOutputElement from '../cotoutputelement/CotOutputElement';
import { FAVORED } from '../datasetentry/DatasetEntry';
import styles from './DatasetEntryElement.module.scss';

interface DatasetEntryElementProps {
  cotData: CotData
  similarityDict: SimilaritiesDict
  selectedSimilarityType: string
  username: string
  visualisationTreshold: number
  anyUpdatePerformed: () => void
}

type SimilaritiesDict = Record<string, SimilarityInfo[]>

const DatasetEntryElement: FC<DatasetEntryElementProps> = (props) => {
  //const [similarities, setSimilarities] = useState<SimilaritiesDict>({})
  //const [similarityType, setSimilarityType] = useState<string>()
  const [bestCotIndex, setBestCotIndex] = useState<number>(props.cotData.generated_cot.findIndex(cotData => cotData['isFavored']))

  const lengths = props.cotData.lengths
  const sentences = props.cotData.sentences

  const similarities = props.similarityDict
  const similarityType = props.selectedSimilarityType

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
    while (blockIndex + 1 < lengths!.length && index >= lengths![blockIndex] + currentLength) {
      currentLength += lengths![blockIndex]
      blockIndex++
    }
    return blockIndex
  }

  // Previously in render method
  const sentenceElementsDict = props.cotData.generated_cot.reduce((aggr, value, index) => {
    aggr[index] = []
    return aggr
  }, {} as SentenceElementDict)

  sentences!.forEach((sentence, index) => {
    const blockIndex = getBlockIndex(index)

    // by contract, sentence can only appear once in entire similarities information
    let similarity
    if (similarityType && similarities) {
      const similaritiesForType = similarities[similarityType]
      similarity = similaritiesForType.find(similarity => similarity.indices?.find(i => i == index) != null)
    }

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
        visualisationTreshold={props.visualisationTreshold}
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
