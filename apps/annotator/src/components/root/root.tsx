import { FC, useEffect, useState } from 'react';
import Slider from 'react-input-slider';
import CotData, { SimilaritiesDict } from '../../dtos/CotData';
import Annotator from '../annotator/annotator';
import Dropdown from '../dropdown/dropdown';
import { post } from '../httpservice';
import Login from '../login/login';
import styles from './root.module.scss';

interface RootProps { }

const Root: FC<RootProps> = () => {
  const [username, setUsername] = useState<string>()
  const [filename, setFilename] = useState('')
  const [allData, setAllData] = useState<any>()
  const [cotData, setCotData] = useState<CotData[]>()
  const [loggedIn, setLoggedIn] = useState(false)
  const [downloadData, setDownloadData] = useState<string>()
  const [sliderX, setSliderX] = useState(0.5)
  const [similarityDicts, setSimilarityDicts] = useState<SimilaritiesDict[]>([])
  const [similarityTypes, setSimilarityTypes] = useState<string[]>()
  const [selectedSimilarityType, setSelectedSimilarityType] = useState<string>("")
  const [lastBackupTime, setLastBackupTime] = useState(0)

  useEffect(() => {
    setupBeforeUnloadListener()

    // Function called on unmount
    return () => backupFileToSession()
  }, [])  // Pass an empty array to run callback on mount only.

  // Setup the `beforeunload` event listener
  const setupBeforeUnloadListener = () => {
    window.addEventListener("beforeunload", (ev) => {
      // ev.preventDefault();
      post('backup', JSON.stringify({ smt: 'smtaaa' }), (data) => {
        console.log("Backup success")
      })
    });
  };

  function onFileRead(filename: string, allData: any, cotData: CotData[]) {
    cotData.forEach(cotData => {
      const cot = cotData.generated_cot.map(cotData => cotData.cot)
      const sentencesForCots = cot.map(cot => cot.split(". ")
        .map((sentence, index, arr) => (index != arr.length - 1) ? sentence + ". " : sentence))

      cotData.sentences = sentencesForCots.flatMap(sentences => sentences)
      cotData.lengths = sentencesForCots.map(sentences => sentences.length)
    })

    setFilename(filename)
    setAllData(allData)
    setCotData(cotData)

    const postData = cotData.map(cotData => {
      return {
        'sentences': cotData.sentences,
        'lengths': cotData.lengths
      }
    })

    post('compareall', postData, (similarityDicts: SimilaritiesDict[]) => {
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
  }

  function onLogin() {
    if (username && username.length > 0 && cotData && cotData.length > 0) {
      setLoggedIn(true)
    }
  }

  function updateExportFile() {
    const downloadData = "data:text/plain;charset=utf-8," + encodeURIComponent(JSON.stringify(allData))
    setDownloadData(downloadData)

    const currentTimeMillis = Date.now()
    if (currentTimeMillis - lastBackupTime > 15_000) {
      backupFileToSession()
      setLastBackupTime(currentTimeMillis)
    }
  }

  function backupFileToSession() {
    const postData = {
      username: username,
      filename: filename,
      filecontent: allData
    }

    console.log(postData)
    post('backup', postData, (data) => {
      console.log("Backup success")
    })
  }

  function logout() {
    setLoggedIn(false)
  }

  return <div className={styles.Root}>
    <div className={styles.HeaderBackground}>
      <div className={styles.Header}>
        <div className={styles.Title}>⚡ThoughtSource Annotator</div>
        <div className={styles.Menu}>
          <a href={downloadData} download="export.json">Download current | </a>
          <Dropdown options={similarityTypes || []} onClick={setSelectedSimilarityType} currentChoice={selectedSimilarityType} />
          <span>Visualisation treshold</span>
          <Slider axis="x" x={sliderX} onChange={({ x }) => setSliderX(x)} xmin={0} xmax={1} xstep={0.05} />
          <div className={styles.User}><div className={styles.InnerUser}><span>{username}</span><span onClick={logout}>Logout</span></div></div>
        </div>
      </div>
    </div>
    {loggedIn ?
      <div>
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
