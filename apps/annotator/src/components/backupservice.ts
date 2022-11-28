export const USERNAME_KEY = 'username'
export const FILE_NAME_KEY = 'filename'
export const FILE_CONTENT_KEY = 'filecontent'

export function backupCurrentData(username: any, filename: string, allData: any, onError: () => void) {
    if (!username || username.length <= 0
        || !filename || filename.length <= 0
        || allData == null) {
        return
    }

    try {
        localStorage.setItem("localJson", JSON.stringify(allData))
        localStorage.setItem("username", username)
        localStorage.setItem("filename", filename)
    } catch (e) {
        onError()
        console.log("Error writing to local storage")
        console.log(e)
    }
}

export function restoreBackup() {
    const localJson = localStorage.getItem("localJson")
    if (localJson == null) {
        return
    }

    const username = localStorage.getItem("username")
    const filename = localStorage.getItem("filename")

    if (username == null || filename == null) {
        return
    }

    return {
        [USERNAME_KEY]: username,
        [FILE_NAME_KEY]: filename,
        [FILE_CONTENT_KEY]: localJson
    }
}