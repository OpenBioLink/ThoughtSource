import { FC, useEffect, useState } from 'react';
import CotData from '../../dtos/CotData';
import { FILE_CONTENT_KEY, FILE_NAME_KEY, get, USERNAME_KEY } from '../httpservice';
import styles from './login.module.scss';

interface LoginProps {
  onUsername: (username: string) => void
  onFileRead: (filename: string, allData: any, cotData: CotData[]) => void
  onLogin: () => void
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
      console.log("Checkin success")
      console.log(data)
    })
  }

  function onFileChange(event: any) {
    const filename = event.target.files[0].name

    const reader = new FileReader()
    reader.readAsText(event.target.files[0], "UTF-8")
    reader.onload = (loadEvent) => onFileRead(filename, JSON.parse(loadEvent.target?.result as any))
  }

  function onFileRead(filename: string, data: any) {
    // Flatmap all dataset entries
    let stateEntries: CotData[] = []
    for (let [i, dataset] of Object.entries(data)) {
      const trainingEntries = (dataset as any)['train']
      const testEntries = (dataset as any)['test']
      const validationEntries = (dataset as any)['validation']
      stateEntries = [...stateEntries, ...trainingEntries, ...testEntries, ...validationEntries]
    }

    // Filter all entries without CoT
    stateEntries = stateEntries.filter(entry => entry.generated_cot?.length > 0)

    // Pass both original and parsed data back to Root
    props.onFileRead(filename, data, stateEntries)
  }

  console.log("trying the session element")
  console.log(state)
  console.log(state ? 'true' : 'false')
  const existingSessionElement = state ? <div onClick={() => {
    props.onUsername(state[USERNAME_KEY])
    onFileRead(state[FILE_NAME_KEY], state[FILE_CONTENT_KEY])
  }}>
    {state[USERNAME_KEY]} {state[FILE_NAME_KEY]}
  </div> : null

  return <div className={styles.Login}>
    <label>File</label>
    <input type="file" onChange={onFileChange} />
    <label>Name</label>
    <input onChange={(event) => props.onUsername(event.target.value)}></input>
    <button onClick={props.onLogin}>Start annotating</button>
    {existingSessionElement}
  </div>
}

export default Login;
