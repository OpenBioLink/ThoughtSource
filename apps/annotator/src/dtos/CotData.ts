export type Annotation = {
    author?: string
    date?: string
    key?: string
    value?: string
    comment?: string
}

export type CotOutput = {
    template_version?: string
    instruction?: string
    cot_trigger?: string
    answers?: [{
        'answer-extraction': string
        answer?: string
    }]
    answer: string
    cot: string
    author?: string
    data?: string
    model?: string
    comment?: string
    annotations?: Annotation[]
    isFavored?: boolean
}

export function findExistingAnnotation(cotOutput: CotOutput, key: string, username: string) {
    return cotOutput.annotations?.find(annotation => annotation.key == key && annotation.author == username)
}

export function annotate(cotOutput: CotOutput, key: string, value: any, author: string, comment: any) {
    const today = new Date().toISOString().slice(0, 10)
    let annotation = findExistingAnnotation(cotOutput, key, author)

    if (!annotation) {
        annotation = {
            key: key,
            value: value,
            comment: comment,
            author: author,
            date: today
        }
        if (!cotOutput.annotations) {
            cotOutput.annotations = [annotation]
        } else {
            cotOutput.annotations?.push(annotation)
        }
    } else {
        annotation.value = value;
        annotation.comment = comment;
        annotation.author = author;
        annotation.date = today;
    }

    return annotation
}

export type SimilaritiesDict = Record<string, SimilarityInfo[]>

export type SimilarityInfo = {
    indices: number[]
    similarity_score: number
    index?: number
}

export type SentenceElementDict = Record<number, SentenceElement[]>

export type SentenceElement = {
    sentence: string,
    similarityIndex?: number,
    similarityScore?: number
}

type CotData = {
    // Properties to interface with ThoughtSource
    id?: string
    question_id?: string
    document_id?: string
    question?: string
    type?: string
    cot_type?: string
    choices?: string[]
    context?: string
    cot?: string[]
    answer: string[]

    generated_cot: CotOutput[]
    feedback?: [string]

    // Internal to annotator frontend
    similarities?: SimilaritiesDict
    sentences?: string[]    // Stores sentences of all generated_cot, seperated by full stops
    lengths?: number[]      // Stores number of sentences for each generated_cot
}

export default CotData;