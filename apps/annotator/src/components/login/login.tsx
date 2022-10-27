import { FC, useEffect } from 'react';
import CotData from '../../dtos/CotData';
import styles from './login.module.scss';

interface LoginProps {
  onUsername: (username: string) => void
  onFileRead: (allData: any, cotData: CotData[]) => void
  onLogin: () => void
}

const Login: FC<LoginProps> = (props) => {

  function makePostRequest() {

  }

  useEffect(() => {
    const requestOptions = {
      method: 'GET',
      credentials: 'include',
      origin: 'http://localhost:3000',
    }

    fetch('http://localhost:5000/checkin', requestOptions as any)
      .then(response => response.json())
      .then((data) => {
        console.log("Success")
        console.log(data)
      })
      .catch(error => {
        console.log("Error request")
        console.log(error)
      })
  }, [])  // Pass an empty array to run callback on mount only.

  function onFileChange(event: any) {
    const reader = new FileReader()
    reader.readAsText(event.target.files[0], "UTF-8")
    reader.onload = onFileRead
  }

  function onFileRead(event: any) {
    // Flatmap all dataset entries
    const data = JSON.parse(event.target.result)
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
    props.onFileRead(data, stateEntries)
  }

  return <div className={styles.Login}>
    <label>File</label>
    <input type="file" onChange={onFileChange} />
    <label>Name</label>
    <input onChange={(event) => props.onUsername(event.target.value)}></input>
    <button onClick={props.onLogin}>Start annotating</button>
  </div>
}

export default Login;
