import { FC, useState } from 'react';
import Slider from 'react-input-slider';
import CotData from '../../dtos/CotData';
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
