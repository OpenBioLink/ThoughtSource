import { FC } from 'react';
import Slider from 'react-input-slider';
import Dropdown from '../dropdown/dropdown';
import styles from './header.module.scss';


interface HeaderProps {
  username?: string
  onLogout: () => void
  tresholdValue: number
  setTresholdValue: (value: number) => void
  downloadData: any
  similarityTypes?: string[]
  selectedSimilarityType: string
  onSelectSimilarityType: (type: string) => void
  isLoggedIn: boolean
}

const Header: FC<HeaderProps> = (props) => {

  return <div className={styles.HeaderBackground}>
    <div className={styles.Header}>
      <div className={styles.Title}>⚡ThoughtSource Annotator</div>
      {props.isLoggedIn ?
        <div className={styles.Menu}>
          <a href={props.downloadData} download="export.json">Download current | </a>
          <Dropdown options={props.similarityTypes || []} onClick={props.onSelectSimilarityType} currentChoice={props.selectedSimilarityType} />
          <span>Visualisation treshold</span>
          <Slider axis="x" x={props.tresholdValue} onChange={({ x }) => props.setTresholdValue(x)} xmin={0} xmax={1} xstep={0.05} />
          <div className={styles.User}><div className={styles.InnerUser}><span>{props.username}</span><span onClick={props.onLogout}>Logout</span></div></div>
        </div> : null
      }
    </div>
  </div>
}

export default Header;
