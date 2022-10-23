import React from "react";
import CotData, { annotate, CotOutput, findExistingAnnotation } from "../../dtos/CotData";
import CotOutputElement from "../cotoutputelement/CotOutputElement";
import { levenshtein } from "../levenshtein";
import styles from './DatasetEntry.module.scss';

//const annotationList = ["Inaccurate", "Too verbose", "Wrong reasoning"]
export const annotationList = ["Incorrect reasoning", "Insufficient knowledge", "Incorrect reading comprehension", "Too verbose"]
export const FAVORED = "preferred"
export const FREETEXT = "freetext"

export type SimilarityInfo = {
    indices: number[]
    similarity_score: number
    index?: number
}

class DatasetEntry extends React.Component<{
    cotData: CotData
    author: string
    anyUpdatePerformed: () => void
}, {
    similarities: SimilarityInfo[]
    bestCotIndex: number
}>
{
    private sentences: string[];
    private lengths: number[];

    constructor(props: any) {
        super(props)

        const answers = this.props.cotData.generated_cot.map(cotData => cotData.cot)
        const splitAnswers = answers.map(answer => answer.split(". ")
            .map((s, index, arr) => {
                if (index - 1 == arr.length) {
                    return s
                }
                return s + ". "
            }))

        this.lengths = splitAnswers.map(answerArray => answerArray.length)
        this.sentences = splitAnswers.flatMap(answerArray => answerArray)

        const bestCotIndex = this.props.cotData.generated_cot.findIndex(cotData => cotData['isFavored'])

        this.state = {
            similarities: [],
            bestCotIndex: bestCotIndex
        }
    }

    componentDidMount(): void {
        const requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sentences: this.sentences,
                lengths: this.lengths
            })
        }

        fetch('http://localhost:5000/textcompare', requestOptions)
            .then(response => response.json())
            .then((data: SimilarityInfo[]) => {
                // TODO debugging things
                //console.log("For " + this.props.cotData.question)
                //console.log(data)
                data.forEach((value, index) => value.index = index)
                this.setState({ similarities: data })
            })
            .catch(error => {
                console.log("Error fetching similarities")
                console.log(error)
            })
    }

    render() {

        const sentenceElementsDict = this.props.cotData.generated_cot.reduce((aggr, value, index) => {
            aggr[index] = []
            return aggr
        }, {} as SentenceElementDict)

        this.sentences.forEach((sentence, index) => {
            const blockIndex = this.getBlockIndex(index)

            // by contract, sentence can only appear once in entire similarities information
            const similarity = this.state.similarities?.find(similarity => similarity.indices?.find(i => i == index) != null)

            const sentenceElement = { sentence: sentence, similarityIndex: similarity?.index, similarityScore: similarity?.similarity_score } as SentenceElement
            sentenceElementsDict[blockIndex].push(sentenceElement)

            // TODO EVEN MORE DEBUG 
            //if (sentence.trim().startsWith("The first step is to understand")) {
            //    console.log("THE FIRST STEP IS TO UNDESTAND")
            //    console.log(sentence)
            //}
        })

        // TODO MORE DEBUG
        //if (this.props.cotData.question?.trim().startsWith("Why are different star")) {
        //    console.log("ANOI")
        //    console.log(this.state.similarities)
        //    console.log(this.sentences)
        //    console.log(sentenceElementsDict)
        //}


        const resultElements = this.props.cotData.generated_cot?.map((cotOutput, index) =>
            <li key={index}>
                <CotOutputElement
                    cotOutput={cotOutput}
                    sentenceElements={sentenceElementsDict[index]}
                    bestCot={this.state.bestCotIndex == index}
                    correctAnswer={this.props.cotData.answer[0]}
                    username={this.props.author}
                    updateBestCot={() => this.updateBestCot(index)}
                    updateExportFile={this.props.anyUpdatePerformed} />
            </li>)

        return (
            <div className={styles.DatasetEntry} >
                <div className={styles.EntryHeader}>
                    <h3>Question</h3>
                    <span>{this.props.cotData.question}</span>
                    <br />
                    <span style={{ fontStyle: "oblique" }}>Correct answer: {this.props.cotData.answer}</span>
                </div>
                <ul className={styles.OutputsContainer}>
                    {resultElements}
                </ul>
            </div>
        )
    }

    private updateBestCot(bestCotIndex: number) {
        this.setState({ bestCotIndex: bestCotIndex })
        this.props.cotData.generated_cot?.forEach((cotOutput, index) => {
            cotOutput.isFavored = bestCotIndex == index
            // find annotation for it
            const annotation = findExistingAnnotation(cotOutput, FAVORED)
        })

        this.props.anyUpdatePerformed()
    }

    private getBlockIndex(index: number) {
        const lengths = this.lengths
        let blockIndex = 0
        let currentLength = 0
        while (blockIndex + 1 < lengths.length && index >= lengths[blockIndex] + currentLength) {
            currentLength += lengths[blockIndex]
            blockIndex++
        }
        return blockIndex
    }
}

export type SentenceElementDict = Record<number, SentenceElement[]>

export type SentenceElement = {
    sentence: string,
    similarityIndex?: number,
    similarityScore?: number
}

class OutputElement extends React.Component<{
    cotOutput: CotOutput
    sentenceElements: SentenceElement[]
    bestCot: boolean
    answer: string
    author: string
    updateBestCot: () => void
    anyUpdatePerformed: () => void
}, {}>
{
    constructor(props: any) {
        super(props)
    }

    render() {
        const annotationInputs = annotationList.map((annotationString, index) => <li key={index}>
            <label><input
                type="checkbox" name="name"
                checked={findExistingAnnotation(this.props.cotOutput, annotationString)?.value == "true"}
                onChange={(e) => { this.onAnnotationClicked(annotationString, e) }} />
                {annotationString}</label>
        </li>)

        // TODO debugging things
        //if (this.props.cotOutput.cot.trim().startsWith("The first step is to understand that the Earth revolves")) {
        //    console.log("HERE WE GO")
        //    console.log(this.props.sentenceElements)
        //}

        const sentenceOutputs = this.props.sentenceElements.map((sentenceElement, index) => {
            const color = this.getSimilarityBackgroundColor(sentenceElement?.similarityIndex)
            return <span style={{ 'backgroundColor': color } as any}>{sentenceElement.sentence}</span>
        })

        const answer = this.props.cotOutput.answers?.find(a => a['answer-extraction'] == "kojima-01")?.answer
        const answerExtractionRegex = /^ *[A-Z][)] */i
        const trimmedAnswer = answer?.replace(answerExtractionRegex, "").trim()
        const distance = levenshtein(trimmedAnswer, this.props.answer.trim())
        const isCorrect = distance <= 1
        const correctnessIcon = isCorrect ?
            <i className="fa-regular fa-circle-check" style={{ color: "green" }}></i>
            : <i className="fa-regular fa-circle-xmark" style={{ color: "#ce1c1c" }}></i>

        const favIcon = this.props.bestCot ? "fa-solid fa-star" : "fa-regular fa-star"
        return (
            <div>
                <div className={styles.Sentences}>
                    {sentenceOutputs}
                </div>
                <div>
                    {correctnessIcon}
                    <span> Answer: {answer?.trim()}</span>
                </div>
                <div className={styles.BestCot}>
                    <i className={favIcon} style={{ fontSize: 20 }} onClick={this.props.updateBestCot} />
                </div>
                <ul className={styles.Annotations}>
                    {annotationInputs}
                    <input onBlur={this.onFreetext} value={findExistingAnnotation(this.props.cotOutput, FREETEXT)?.comment} title="freetext"></input>
                </ul>
            </div>
        )
    }

    private onFreetext = (event: any) => {
        const text = event.target.value
        annotate(this.props.cotOutput, FREETEXT, null, this.props.author, text)

        this.props.anyUpdatePerformed()
    }

    private onAnnotationClicked = (key: string, event: any) => {
        const value = `${event.target.checked}`
        annotate(this.props.cotOutput, key, value, this.props.author, null)

        this.props.anyUpdatePerformed()
    }

    private getSimilarityBackgroundColor(similarityIndex?: number) {
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
}

export default DatasetEntry