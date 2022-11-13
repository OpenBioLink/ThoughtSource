import { FC, useState } from 'react';
import CotData, { annotate, CotOutput, findExistingAnnotation, SentenceElement, SentenceElementDict, SimilarityInfo } from '../../dtos/CotData';
import CotOutputElement from '../cotoutputelement/CotOutputElement';
import styles from './DatasetEntryElement.module.scss';

export const FAVORED = "preferred"

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
  const [bestCotIndex, setBestCotIndex] = useState<number>(props.cotData.generated_cot.findIndex(cotData => findExistingAnnotation(cotData, FAVORED, props.username)?.value))

  const lengths = props.cotData.lengths
  const sentences = props.cotData.sentences

  const similarities = props.similarityDict
  const similarityType = props.selectedSimilarityType

  // Previously methods of class
  function updateBestCot(bestCotIndex: number) {
    setBestCotIndex(bestCotIndex)

    // Update each CotOutput's 'favored' annotation
    props.cotData.generated_cot?.forEach((cotOutput, index) => {
      const isBest = bestCotIndex == index
      annotate(cotOutput, FAVORED, isBest, props.username, null)
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

  const goldstandardIndex = (props.cotData.cot && props.cotData.cot.length > 0) ? props.cotData.generated_cot.length : -1
  if (goldstandardIndex >= 0) {
    sentenceElementsDict[props.cotData.generated_cot.length] = []
  }

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

  let resultElements = props.cotData.generated_cot?.map((cotOutput, index) => (
    <CotOutputElement
      key={props.cotData.id + "/" + index}
      cotOutput={cotOutput}
      sentenceElements={sentenceElementsDict[index]}
      bestCot={bestCotIndex == index}
      correctAnswer={props.cotData.answer[0]}
      username={props.username}
      visualisationTreshold={props.visualisationTreshold}
      updateBestCot={() => updateBestCot(index)}
      updateExportFile={props.anyUpdatePerformed} />
  ))

  if (goldstandardIndex >= 0) {
    resultElements = [<CotOutputElement
      isGoldstandard={true}
      key={props.cotData.id + "/" + goldstandardIndex}
      cotOutput={{} as CotOutput}
      sentenceElements={sentenceElementsDict[goldstandardIndex]}
      bestCot={false}
      correctAnswer={props.cotData.answer[0]}
      username={props.username}
      visualisationTreshold={props.visualisationTreshold}
      updateBestCot={() => { }}
      updateExportFile={props.anyUpdatePerformed} />,
    ...resultElements]
  }

  const answerElements = props.cotData.choices?.map(choice => {
    const isCorrectAnswer = choice == props.cotData.answer[0]
    return <li className={isCorrectAnswer ? styles.CorrectAnswer : ''}>{choice}</li>
  })

  return <div className={styles.DatasetEntryElement}>
    <div className={styles.EntryHeader}>
      <h3>Question {props.cotData.id} ({props.cotData.subsetType})</h3>
      <span>{props.cotData.question}</span>
      <ol>{answerElements}</ol>
    </div>
    <ul className={styles.OutputsContainer}>
      {resultElements}
    </ul>
  </div>
}

export default DatasetEntryElement;
