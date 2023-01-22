import { FC } from 'react';
import { annotate, CotOutput, findExistingAnnotation, SentenceElement } from '../../dtos/CotData';
import styles from './CotOutputElement.module.scss';

const COMMENT = "comment"
const annotationList = ["Incorrect reasoning", "Insufficient knowledge", "Incorrect reading comprehension", "Too verbose"]
const colors_v5 = ['#23DCAE', '#536AD8', '#AE66E1', '#DEC721', '#76D965']

interface CotOutputElementProps {
  cotOutput: CotOutput
  sentenceElements: SentenceElement[]
  bestCot: boolean
  correctAnswer: string
  username: string
  visualisationTreshold: number
  updateBestCot: () => void
  updateExportFile: () => void
  isGoldstandard?: boolean
}

const CotOutputElement: FC<CotOutputElementProps> = (props) => {

  const sentenceOutputs = props.sentenceElements.map((sentenceElement, index) => {
    const color = getSimilarityBackgroundColor(sentenceElement?.similarityIndex)
    const style = sentenceElement?.similarityScore != null && sentenceElement.similarityScore > props.visualisationTreshold ?
      { 'backgroundColor': color } : {}
    const sentenceText = sentenceElement.sentence.endsWith(" ") ? sentenceElement.sentence : sentenceElement.sentence + " "
    return <span style={style as any}>{sentenceText}</span>
  })

  if (props.isGoldstandard) {
    return <div className={styles.CotOutputElement}>
      <p style={{ fontStyle: 'oblique', textAlign: 'center' }}>Gold standard CoT</p>
      <div>
        {sentenceOutputs}
      </div>
    </div>
  }

  function onFreetext(event: any) {
    const text = event.target.value
    annotate(props.cotOutput!, COMMENT, text, props.username, null)

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

    if (similarityIndex >= 0 && similarityIndex < colors_v5.length) {
      return colors_v5[similarityIndex]
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

  // TODO handling for multiple answer extractions needed? if so, implement toggle similar to similarity measure at root level and pass it's value down to CotOutputElement.
  //const answerEntry = props.cotOutput.answers?.find(a => a['answer-extraction'] == "kojima-01")
  const answerEntry = props.cotOutput.answers?.at(0)
  const answer = answerEntry?.answer
  let correctnessIcon
  if (answerEntry?.correct_answer == null) {
    correctnessIcon = <i className="fa-regular fa-question-circle" style={{ color: "#777777" }}></i>
  } else {
    const isCorrect = answerEntry?.correct_answer == true
    correctnessIcon = isCorrect ?
      <i className="fa-regular fa-circle-check" style={{ color: "green" }}></i>
      : <i className="fa-regular fa-circle-xmark" style={{ color: "#ce1c1c" }}></i>
  }


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
    </ul>
    <textarea onBlur={onFreetext} defaultValue={findExistingAnnotation(props.cotOutput, COMMENT, props.username)?.value} title="freetext" className={styles.Comment}></textarea >
  </div>
}

export default CotOutputElement;
