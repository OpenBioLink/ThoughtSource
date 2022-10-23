import { FC, useState } from 'react';
import CotData from '../../dtos/CotData';
import Annotator from '../annotator/annotator';
import Login from '../login/login';
import styles from './root.module.scss';

interface RootProps { }

const Root: FC<RootProps> = () => {
  const [username, setUsername] = useState<string>()
  const [allData, setAllData] = useState<any>()
  const [cotData, setCotData] = useState<CotData[]>()
  const [loggedIn, setLoggedIn] = useState(false)
  const [downloadData, setDownloadData] = useState<string>()

  function onFileRead(allData: any, cotData: CotData[]) {
    setAllData(allData)
    setCotData(cotData)
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
          <span>ThoughtSource Annotator</span>
          <a href={downloadData} download="export.json">Download current</a>
        </div>
        <Annotator
          username={username as string}
          cotData={cotData as CotData[]}
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
