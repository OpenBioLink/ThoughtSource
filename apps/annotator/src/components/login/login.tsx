import { FC, useEffect, useState } from 'react';
import CotData from '../../dtos/CotData';
import { FILE_CONTENT_KEY, FILE_NAME_KEY, get, USERNAME_KEY } from '../httpservice';
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
    reader.onload = (loadEvent) => onFileRead(filename, JSON.parse(loadEvent.target?.result as any), false)
  }

  function onFileRead(filename: string, data: any, startAnnotating: boolean) {
    // Flatmap all dataset entries
    let stateEntries: CotData[] = []

    // Loop through datasets
    for (let [i, dataset] of Object.entries(data)) {

      // Each dataset is expected to have subsets such as "train" or "test" - iterate through these
      for (let setType of Object.keys(dataset as any)) {
        console.log(`Reading ${setType} of ${dataset}`)
        const entries = (dataset as any)[setType]
        stateEntries = [...stateEntries, ...entries]
      }
      //const trainingEntries = (dataset as any)['train']
      //const testEntries = (dataset as any)['test']
      //const validationEntries = (dataset as any)['validation']
      //stateEntries = [...stateEntries, ...trainingEntries, ...testEntries, ...validationEntries]
    }

    // Filter all entries without CoT
    stateEntries = stateEntries.filter(entry => entry.generated_cot?.length > 0)

    // Pass both original and parsed data back to Root
    props.onFileRead(filename, data, stateEntries, startAnnotating)
  }

  const existingSessionElement = state ? <div className={styles.RestoreSession}>
    <h5>
      Restore previous session
    </h5>
    <div onClick={() => {
      props.onUsername(state[USERNAME_KEY])
      onFileRead(state[FILE_NAME_KEY], state[FILE_CONTENT_KEY], true)
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
