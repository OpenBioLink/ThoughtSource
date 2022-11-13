import CotData from "../dtos/CotData"

export function parseCotData(filename: string, data: any, startAnnotating: boolean,
    onSuccess: (filename: string, allData: any, cotData: CotData[], startAnnotating: boolean) => void
) {
    // Flatmap all dataset entries
    let stateEntries: CotData[] = []

    // Loop through datasets
    for (let [i, dataset] of Object.entries(data)) {

        // Each dataset is expected to have subsets such as "train" or "test" - iterate through these
        for (let setType of Object.keys(dataset as any)) {
            const entries = (dataset as any)[setType] as CotData[]
            if (entries == null || entries.length == 0) {
                console.log(`Skipping ${setType} - no valid entries`)
                continue
            }

            console.log(`Reading ${setType} of ${dataset}`)
            entries.forEach(cotData => cotData.subsetType = setType)
            stateEntries = [...stateEntries, ...entries]
        }
    }

    // Filter all entries without CoT
    stateEntries = stateEntries.filter(entry => entry.generated_cot?.length > 0)

    // Sort entries by id and subset type
    stateEntries
        .sort((a: CotData, b: CotData) => {
            if (a.subsetType! != b.subsetType) {
                return a.subsetType!.localeCompare(b.subsetType!)
            }

            const aId = parseInt(a.id!)
            const bId = parseInt(b.id!)

            if (isNaN(aId) || isNaN(bId)) {
                if (!isNaN(aId)) {
                    return -1
                }
                if (!isNaN(bId)) {
                    return 1
                }

                return a.id!.localeCompare(b.id!)
            }
            return aId - bId
        })

    // Pass both original and parsed data back to Root
    onSuccess(filename, data, stateEntries, startAnnotating)
}