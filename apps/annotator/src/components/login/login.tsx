import { FC, useEffect, useState } from 'react';
import CotData from '../../dtos/CotData';
import { restoreBackup } from '../backupservice';
import { FILE_CONTENT_KEY, FILE_NAME_KEY, USERNAME_KEY } from '../httpservice';
import { parseCotData } from '../readfileservice';
import styles from './login.module.scss';

interface LoginProps {
  onUsername: (username: string) => void
  onFileRead: (filename: string, allData: any, cotData: CotData[], startAnnotating: boolean) => void
  onLogin: () => void
  onError: (message: string) => void
  username?: string
  hasCotDataLoaded: boolean
}

const Login: FC<LoginProps> = (props) => {

  const [state, setState] = useState<any>()

  useEffect(() => {
    const backupData = restoreBackup()
    if (!backupData) {
      return
    }

    setState({
      [USERNAME_KEY]: backupData[USERNAME_KEY],
      [FILE_NAME_KEY]: backupData[FILE_NAME_KEY],
      [FILE_CONTENT_KEY]: backupData[FILE_CONTENT_KEY]
    })
  }, [])  // Pass an empty array to run callback on mount only.

  function onFileChange(event: any) {
    props.onError("")
    const filename = event.target.files[0].name

    const reader = new FileReader()
    reader.readAsText(event.target.files[0], "UTF-8")
    reader.onload = (loadEvent) => parseCotData(filename, loadEvent.target?.result as any, false, props.onFileRead, uploadedFileInvalid)
  }

  function uploadedFileInvalid() {
    props.onError("Uploaded file not in valid format - please conform refer to sample file at https://github.com/OpenBioLink/ThoughtSource")
  }

  function localStorageInvalid() {
    props.onError("Technical error: File in localstorage not of valid format. Data stored in'localJson' in browser application cache.")
  }


  const existingSessionElement = state ? <div className={styles.RestoreSession}>
    <h5>
      Restore previous session
    </h5>
    <div onClick={() => {
      props.onError("")
      props.onUsername(state[USERNAME_KEY])
      parseCotData(state[FILE_NAME_KEY], state[FILE_CONTENT_KEY], true, props.onFileRead, localStorageInvalid)
    }}>
      {state[USERNAME_KEY]} | {state[FILE_NAME_KEY]}
    </div>
  </div> : null

  return <div className={styles.Login}>
    <h5>Enter your name</h5>
    <input placeholder='Author' onChange={(event) => props.onUsername(event.target.value)}></input>
    <input type="file" onChange={onFileChange} />
    <button onClick={props.onLogin} disabled={props.username == null || props.username.length == 0 || !props.hasCotDataLoaded}>Start annotating</button>
    {existingSessionElement}
  </div>
}

export default Login;
