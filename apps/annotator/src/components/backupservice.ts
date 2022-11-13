export const USERNAME_KEY = 'username'
export const FILE_NAME_KEY = 'filename'
export const FILE_CONTENT_KEY = 'filecontent'

export function backupCurrentData(username?: string, filename?: string, allData?: any) {
    if (!username || username.length <= 0
        || !filename || filename.length <= 0
        || allData == null) {
        return
    }

    localStorage.setItem("localJson", JSON.stringify(allData))
    localStorage.setItem("username", username)
    localStorage.setItem("filename", filename)

    //const postData = {
    //  username: username,
    //  filename: filename,
    //  filecontent: allData
    //}

    //post('backup', postData, (data) => {
    //  console.log("Backup success")
    //})
}

export function restoreBackup() {
    const localJson = localStorage.getItem("localJson")
    if (localJson == null) {
        return
    }

    const allData = JSON.parse(localJson)
    const username = localStorage.getItem("username")
    const filename = localStorage.getItem("filename")

    if (username == null || filename == null) {
        return
    }

    return {
        [USERNAME_KEY]: username,
        [FILE_NAME_KEY]: filename,
        [FILE_CONTENT_KEY]: allData
    }
}