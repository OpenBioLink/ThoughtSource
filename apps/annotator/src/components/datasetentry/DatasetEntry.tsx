
//const annotationList = ["Inaccurate", "Too verbose", "Wrong reasoning"]
export const annotationList = ["Incorrect reasoning", "Insufficient knowledge", "Incorrect reading comprehension", "Too verbose"]
export const FAVORED = "preferred"
export const FREETEXT = "freetext"

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