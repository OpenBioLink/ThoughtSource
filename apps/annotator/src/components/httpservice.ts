export const SERVER_URL = 'http://localhost:5000'
export const APP_URL = 'http://localhost:3000'

export const USERNAME_KEY = 'username'
export const FILE_NAME_KEY = 'filename'
export const FILE_CONTENT_KEY = 'filecontent'

export function get(endpoint: string, onSuccess: (data: any) => void) {
  const requestOptions = {
    method: 'GET',
    credentials: 'include',
    origin: APP_URL,
  }

  sendRequest(endpoint, requestOptions, onSuccess)
}

export function post(endpoint: string, postData: any, onSuccess: (data: any) => void) {
  const requestOptions = {
    method: 'POST',
    credentials: 'include',
    origin: APP_URL,
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(postData)
  }

  sendRequest(endpoint, requestOptions, onSuccess)
}

function sendRequest(endpoint: string, requestOptions: any, onSuccess: (data: any) => void) {
  fetch(`${SERVER_URL}/${endpoint}`, requestOptions)
    .then(response => response.json())
    .then(onSuccess)
    .catch(error => {
      const requestMethod = requestOptions['method']
      console.log(`Error processing ${requestMethod} request to ${endpoint}`)
      console.log(error)
    })
}
