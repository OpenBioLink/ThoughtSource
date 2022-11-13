import { FC, useEffect, useState } from 'react';
import CotData from '../../dtos/CotData';
import { FILE_CONTENT_KEY, FILE_NAME_KEY, get, USERNAME_KEY } from '../httpservice';
import { parseCotData } from '../readfileservice';
import styles from './login.module.scss';

interface LoginProps {
  onUsername: (username: string) => void
  onFileRead: (filename: string, allData: any, cotData: CotData[], startAnnotating: boolean) => void
  onLogin: () => void
  username?: string
  hasCotDataLoaded: boolean
}

const Login: FC<LoginProps> = (props) => {

  const [state, setState] = useState<any>()

  useEffect(() => {
    performServerCheckin()
  }, [])  // Pass an empty array to run callback on mount only.

  function performServerCheckin() {
    get('checkin', (data) => {
      setState({
        [USERNAME_KEY]: data[USERNAME_KEY],
        [FILE_NAME_KEY]: data[FILE_NAME_KEY],
        [FILE_CONTENT_KEY]: data[FILE_CONTENT_KEY]
      })
    })
  }

  function onFileChange(event: any) {
    const filename = event.target.files[0].name

    const reader = new FileReader()
    reader.readAsText(event.target.files[0], "UTF-8")
    reader.onload = (loadEvent) => parseCotData(filename, JSON.parse(loadEvent.target?.result as any), false, props.onFileRead)
  }

  const existingSessionElement = state ? <div className={styles.RestoreSession}>
    <h5>
      Restore previous session
    </h5>
    <div onClick={() => {
      props.onUsername(state[USERNAME_KEY])
      parseCotData(state[FILE_NAME_KEY], state[FILE_CONTENT_KEY], true, props.onFileRead)
    }}>
      {state[USERNAME_KEY]} | {state[FILE_NAME_KEY]}
    </div>
  </div> : null

  return <div className={styles.Login}>
    <h5>Login</h5>
    <input placeholder='Author' onChange={(event) => props.onUsername(event.target.value)}></input>
    <input type="file" onChange={onFileChange} />
    <button onClick={props.onLogin} disabled={props.username == null || props.username.length == 0 || !props.hasCotDataLoaded}>Start annotating</button>
    {existingSessionElement}
  </div>
}

export default Login;
