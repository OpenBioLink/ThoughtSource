import { FC, useState } from 'react';
import Slider from 'react-input-slider';
import CotData, { SimilaritiesDict } from '../../dtos/CotData';
import Annotator from '../annotator/annotator';
import Dropdown from '../dropdown/dropdown';
import Login from '../login/login';
import styles from './root.module.scss';

interface RootProps { }

const Root: FC<RootProps> = () => {
  const [username, setUsername] = useState<string>()
  const [allData, setAllData] = useState<any>()
  const [cotData, setCotData] = useState<CotData[]>()
  const [loggedIn, setLoggedIn] = useState(false)
  const [downloadData, setDownloadData] = useState<string>()

  const [sliderX, setSliderX] = useState(0.5)


  const [similarityDicts, setSimilarityDicts] = useState<SimilaritiesDict[]>([])
  const [similarityTypes, setSimilarityTypes] = useState<string[]>()
  const [selectedSimilarityType, setSelectedSimilarityType] = useState<string>("")


  function onFileRead(allData: any, cotData: CotData[]) {
    cotData.forEach(cotData => {
      const cot = cotData.generated_cot.map(cotData => cotData.cot)
      const sentencesForCots = cot.map(cot => cot.split(". ")
        .map((sentence, index, arr) => (index != arr.length - 1) ? sentence + ". " : sentence))

      cotData.sentences = sentencesForCots.flatMap(sentences => sentences)
      cotData.lengths = sentencesForCots.map(sentences => sentences.length)
    })

    setAllData(allData)
    setCotData(cotData)

    const postData = cotData.map(cotData => {
      return {
        'sentences': cotData.sentences,
        'lengths': cotData.lengths
      }
    })
    const requestOptions = {
      method: 'POST',
      credentials: 'include',
      origin: 'http://localhost:3000',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        entries: postData,
        username: "TODO testuser"
      })
    } as any

    fetch('http://localhost:5000/compareall', requestOptions)
      .then(response => response.json())
      .then((similarityDicts: SimilaritiesDict[]) => {
        // Store index of similarity to visualise colour later
        similarityDicts.forEach(similarityDict => {
          for (let [key, similarityInfos] of Object.entries(similarityDict)) {
            // TODO does index exist already..?
            similarityInfos.forEach((value, index) => value.index = index)
          }
        })
        setSimilarityDicts(similarityDicts)

        if (similarityDicts != null && similarityDicts.length > 0) {
          const keyNames = Object.keys(similarityDicts[0])
          setSimilarityTypes(keyNames)
          // Initially show similarities for first algorithm
          const firstKeyName = keyNames.at(0) as string
          setSelectedSimilarityType(firstKeyName)
        }
      })
      .catch(error => {
        console.log("Error fetching similarities")
        console.log(error)
      })
  }

  function onLogin() {
    if (username && username.length > 0 && cotData && cotData.length > 0) {
      setLoggedIn(true)
    }
  }

  function updateExportFile() {
    const downloadData = "data:text/plain;charset=utf-8," + encodeURIComponent(JSON.stringify(allData))
    setDownloadData(downloadData)
  }

  return <div className={styles.Root}>
    {loggedIn ?
      <div>
        <div className={styles.Header}>
          <span style={{ color: 'white' }}>ThoughtSource Annotator | </span>
          <a href={downloadData} download="export.json" style={{ color: 'white' }}>Download current | </a>
          <Dropdown options={{

          }} />
          <span style={{ color: 'white' }}>Visualisation treshold</span>
          <Slider axis="x" x={sliderX} onChange={({ x }) => setSliderX(x)} xmin={0} xmax={1} xstep={0.05} />
        </div>
        <Annotator
          username={username as string}
          visualisationTreshold={sliderX}
          cotData={cotData as CotData[]}
          similarityDicts={similarityDicts}
          selectedSimilarityType={selectedSimilarityType}
          anyUpdatePerformed={updateExportFile} />
      </div>
      :
      <Login
        onUsername={setUsername}
        onFileRead={onFileRead}
        onLogin={onLogin} />
    }
  </div>
}

export default Root;
