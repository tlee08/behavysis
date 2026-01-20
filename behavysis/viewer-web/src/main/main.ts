import { app, BrowserWindow, ipcMain, dialog } from "electron";
import path from "path";
import fs from "fs";

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, "../preload/preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false, // Required for local video file access via file://
    },
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("open-file-dialog", async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [
      { name: "Config Files", extensions: ["json", "yaml"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (canceled) {
    return null;
  } else {
    return filePaths[0];
  }
});

ipcMain.handle("read-file", async (_, filePath: string) => {
  try {
    const data = fs.readFileSync(filePath);
    return data.buffer.slice(
      data.byteOffset,
      data.byteOffset + data.byteLength,
    );
  } catch (error) {
    console.error("Error reading file:", error);
    throw error;
  }
});

ipcMain.handle(
  "write-file",
  async (_, filePath: string, content: Uint8Array) => {
    try {
      fs.writeFileSync(filePath, content);
      return true;
    } catch (error) {
      console.error("Error writing file:", error);
      throw error;
    }
  },
);

ipcMain.handle("get-dirname", async (_, filePath: string) => {
  return path.dirname(filePath);
});

ipcMain.handle("join-path", async (_, ...parts: string[]) => {
  return path.join(...parts);
});

ipcMain.handle("path-exists", async (_, filePath: string) => {
  return fs.existsSync(filePath);
});

ipcMain.handle("get-basename", async (_, filePath: string) => {
  return path.basename(filePath, path.extname(filePath));
});
