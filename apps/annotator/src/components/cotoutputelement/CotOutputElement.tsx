import { FC } from 'react';
import { annotate, CotOutput, findExistingAnnotation, SentenceElement } from '../../dtos/CotData';
import { annotationList, FREETEXT } from '../datasetentry/DatasetEntry';
import { levenshtein } from '../levenshtein';
import styles from './CotOutputElement.module.scss';

interface CotOutputElementProps {
  cotOutput: CotOutput
  sentenceElements: SentenceElement[]
  bestCot: boolean
  correctAnswer: string
  username: string
  visualisationTreshold: number
  updateBestCot: () => void
  updateExportFile: () => void
}

const CotOutputElement: FC<CotOutputElementProps> = (props) => {

  function onFreetext(event: any) {
    const text = event.target.value
    annotate(props.cotOutput, FREETEXT, null, props.username, text)

    props.updateExportFile()
  }

  function onAnnotationClicked(key: string, event: any) {
    const value = `${event.target.checked}`
    annotate(props.cotOutput, key, value, props.username, null)

    props.updateExportFile()
  }

  function getSimilarityBackgroundColor(similarityIndex?: number) {
    if (similarityIndex == undefined) {
      return null
    }

    const colors_bright = ['#FBFA30', '#3CFA72', '#32B5FF']
    const colors = ['#20729E', '#249945', '#7d2cc7']
    if (similarityIndex >= 0 && similarityIndex < colors.length) {
      return colors_bright[similarityIndex]
    }
    return null
  }

  const annotationInputs = annotationList.map((annotationString, index) => <li>
    <label><input
      type="checkbox" name="name"
      checked={findExistingAnnotation(props.cotOutput, annotationString, props.username)?.value == "true"}
      onChange={(e) => { onAnnotationClicked(annotationString, e) }} />
      {annotationString}</label>
  </li>)

  const sentenceOutputs = props.sentenceElements.map((sentenceElement, index) => {
    const color = getSimilarityBackgroundColor(sentenceElement?.similarityIndex)
    const style = sentenceElement?.similarityScore != null && sentenceElement.similarityScore > props.visualisationTreshold ?
      { 'backgroundColor': color } : {}
    return <span style={style as any}>{sentenceElement.sentence}</span>
  })

  const answer = props.cotOutput.answers?.find(a => a['answer-extraction'] == "kojima-01")?.answer
  const answerExtractionRegex = /^ *[A-Z][)] */i
  const trimmedAnswer = answer?.replace(answerExtractionRegex, "").trim()
  const distance = levenshtein(trimmedAnswer, props.correctAnswer.trim())
  const isCorrect = distance <= 1
  const correctnessIcon = isCorrect ?
    <i className="fa-regular fa-circle-check" style={{ color: "green" }}></i>
    : <i className="fa-regular fa-circle-xmark" style={{ color: "#ce1c1c" }}></i>

  const favIcon = props.bestCot ? "fa-solid fa-star" : "fa-regular fa-star"


  return <div className={styles.CotOutputElement}>
    <div>
      {sentenceOutputs}
    </div>
    <div>
      {correctnessIcon}
      <span> Answer: {answer?.trim()}</span>
    </div>
    <div className={styles.BestCot}>
      <i className={favIcon} style={{ fontSize: 20 }} onClick={props.updateBestCot} />
    </div>
    <ul className={styles.Annotations}>
      {annotationInputs}
      <input onBlur={onFreetext} value={findExistingAnnotation(props.cotOutput, FREETEXT, props.username)?.comment} title="freetext"></input>
    </ul>  </div>
}

export default CotOutputElement;
