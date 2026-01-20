import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('electronAPI', {
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  readFile: (filePath: string) => ipcRenderer.invoke('read-file', filePath),
  writeFile: (filePath: string, content: Uint8Array) => ipcRenderer.invoke('write-file', filePath, content),
  getDirname: (filePath: string) => ipcRenderer.invoke('get-dirname', filePath),
  joinPath: (...parts: string[]) => ipcRenderer.invoke('join-path', ...parts),
  pathExists: (filePath: string) => ipcRenderer.invoke('path-exists', filePath),
  getBasename: (filePath: string) => ipcRenderer.invoke('get-basename', filePath),
})
