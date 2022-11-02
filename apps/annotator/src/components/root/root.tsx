import { FC, useEffect, useState } from 'react';
import CotData, { SimilaritiesDict } from '../../dtos/CotData';
import Annotator from '../annotator/annotator';
import Header from '../header/header';
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
  const [similarityDicts, setSimilarityDicts] = useState<SimilaritiesDict[]>([])
  const [similarityTypes, setSimilarityTypes] = useState<string[]>()
  const [selectedSimilarityType, setSelectedSimilarityType] = useState<string>("")
  const [tresholdValue, setTresholdValue] = useState(0.5)
  const [lastBackupTime, setLastBackupTime] = useState(0)

  useEffect(() => {
    setupBeforeUnloadListener()

    // Function called on unmount
    return () => backupFileToSession()
  }, [])  // Pass an empty array to run callback on mount only.

  // Setup the `beforeunload` event listener
  // TODO here but better - hier geht momentan nur der OPTIONS request durch...
  const setupBeforeUnloadListener = () => {
    window.addEventListener("beforeunload", (ev) => {
      // ev.preventDefault();
      backupFileToSession()
    });
  };

  function onFileRead(filename: string, allData: any, cotData: CotData[], startAnnotating: boolean) {
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

    if (startAnnotating) {
      onLogin()
    }
  }

  function onLogin() {
    setLoggedIn(true)
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
    <Header
      username={username}
      onLogout={logout}
      tresholdValue={tresholdValue}
      setTresholdValue={setTresholdValue}
      downloadData={downloadData}
      similarityTypes={similarityTypes}
      selectedSimilarityType={selectedSimilarityType}
      onSelectSimilarityType={setSelectedSimilarityType}
      isLoggedIn={loggedIn}
    />
    <div className={styles.Main}>
      {loggedIn ?
        <Annotator
          username={username as string}
          visualisationTreshold={tresholdValue}
          cotData={cotData as CotData[]}
          similarityDicts={similarityDicts}
          selectedSimilarityType={selectedSimilarityType}
          anyUpdatePerformed={updateExportFile} />
        :
        <Login
          onUsername={setUsername}
          onFileRead={onFileRead}
          onLogin={onLogin}
          username={username}
          hasCotDataLoaded={cotData != null && cotData.length > 0} />
      }
    </div>
  </div>
}

export default Root;
