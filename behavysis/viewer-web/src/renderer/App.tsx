import React, { useState, useEffect, useCallback } from "react";
import {
  Box,
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from "@mui/material";
import Grid from "@mui/material/Grid";
import VideoPlayer from "./components/VideoPlayer";
import TimelineGraph from "./components/TimelineGraph";
import BoutsList from "./components/BoutsList";
import BoutInspector from "./components/BoutInspector";
import { Bout, BoutsData } from "./models/Bout";
import { KeypointsModel } from "./models/KeypointsModel";
import { Table } from "apache-arrow";
import initParquet, * as parquet from "parquet-wasm";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#1976d2" },
  },
});

declare global {
  interface Window {
    electronAPI: {
      openFileDialog: () => Promise<string | null>;
      readFile: (filePath: string) => Promise<ArrayBuffer>;
      writeFile: (filePath: string, content: Uint8Array) => Promise<boolean>;
      getDirname: (filePath: string) => Promise<string>;
      joinPath: (...parts: string[]) => Promise<string>;
      pathExists: (filePath: string) => Promise<boolean>;
      getBasename: (filePath: string) => Promise<string>;
    };
  }
}

const App: React.FC = () => {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [boutsData, setBoutsData] = useState<BoutsData | null>(null);
  const [selectedBoutIndex, setSelectedBoutIndex] = useState(-1);
  const [keypointsModel, setKeypointsModel] = useState<KeypointsModel | null>(
    null,
  );
  const [fps, setFps] = useState(30);
  const [showKeypoints, setShowKeypoints] = useState(true);
  const [configPath, setConfigPath] = useState<string | null>(null);
  const [behavsFp, setBehavsFp] = useState<string | null>(null);

  const handleOpenConfig = async () => {
    const fp = await window.electronAPI.openFileDialog();
    if (!fp) return;
    setConfigPath(fp);

    // Initialize parquet-wasm
    await initParquet();

    const dir = await window.electronAPI.getDirname(fp);
    const rootDir = await window.electronAPI.getDirname(dir);
    const name = await window.electronAPI.getBasename(fp);

    // Load Config JSON
    const configBuf = await window.electronAPI.readFile(fp);
    const config = JSON.parse(new TextDecoder().decode(configBuf));
    if (config.auto?.formatted_vid?.fps) setFps(config.auto.formatted_vid.fps);

    // Load Video
    const vidFp = await window.electronAPI.joinPath(
      rootDir,
      "2_formatted_vid",
      `${name}.mp4`,
    );
    setVideoSrc(`file://${vidFp}`);

    // Load Keypoints (Parquet)
    const dlcFp = await window.electronAPI.joinPath(
      rootDir,
      "4_preprocessed",
      `${name}.parquet`,
    );
    if (await window.electronAPI.pathExists(dlcFp)) {
      const dlcBuf = await window.electronAPI.readFile(dlcFp);
      const table = new Table(
        parquet.readParquet(new Uint8Array(dlcBuf)) as any,
      );
      const model = new KeypointsModel();
      model.load(table, {
        colour_level: "individuals",
        pcutoff: config.user?.evaluate_vid?.pcutoff || 0.9,
        radius: config.user?.evaluate_vid?.radius || 5,
        cmap: config.user?.evaluate_vid?.cmap || "viridis",
      });
      setKeypointsModel(model);
    }

    // Load Bouts (Parquet or JSON)
    const bFp = await window.electronAPI.joinPath(
      rootDir,
      "7_scored_behavs",
      `${name}.parquet`,
    );
    setBehavsFp(bFp);
    if (await window.electronAPI.pathExists(bFp)) {
      const bBuf = await window.electronAPI.readFile(bFp);
      const table = new Table(parquet.readParquet(new Uint8Array(bBuf)) as any);

      // Convert Arrow Table to BoutsData
      const bouts: Bout[] = [];
      for (let i = 0; i < table.numRows; i++) {
        const row = table.get(i);
        if (row) {
          bouts.push({
            start: Number(row.start),
            stop: Number(row.stop),
            dur: Number(row.dur),
            behav: String(row.behav),
            actual: Number(row.actual),
            user_defined: JSON.parse(String(row.user_defined || "{}")),
          });
        }
      }
      setBoutsData({
        start: bouts.length > 0 ? Math.min(...bouts.map((b) => b.start)) : 0,
        stop: bouts.length > 0 ? Math.max(...bouts.map((b) => b.stop)) : 0,
        bouts,
        bouts_struct: [], // To be populated from unique behaviors
      });
    }
  };

  const handleSave = async () => {
    if (!behavsFp || !boutsData) return;
    try {
      const jsonContent = JSON.stringify(boutsData, null, 2);
      const uint8 = new TextEncoder().encode(jsonContent);
      const success = await window.electronAPI.writeFile(
        behavsFp.replace(".parquet", ".json"),
        uint8,
      );
      if (success) {
        alert("Data saved successfully!");
      }
    } catch (err) {
      console.error("Failed to save:", err);
      alert("Failed to save data");
    }
  };

  const updateBout = (updates: Partial<Bout>) => {
    if (boutsData && selectedBoutIndex >= 0) {
      const newBouts = [...boutsData.bouts];
      newBouts[selectedBoutIndex] = {
        ...newBouts[selectedBoutIndex],
        ...updates,
      };
      setBoutsData({ ...boutsData, bouts: newBouts });
    }
  };

  const handleSelectBout = (index: number) => {
    setSelectedBoutIndex(index);
    if (boutsData) {
      const bout = boutsData.bouts[index];
      setCurrentTime(Math.max(0, bout.start / fps - 1));
    }
  };

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.code === "KeyK") {
        setShowKeypoints((prev) => !prev);
      } else if (e.code === "Digit1") {
        updateBout({ actual: 1 });
      } else if (e.code === "Digit2") {
        updateBout({ actual: 0 });
      } else if (e.code === "Digit3") {
        updateBout({ actual: -1 });
      }
    },
    [boutsData, selectedBoutIndex],
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: "flex", flexDirection: "column", height: "100vh" }}>
        <AppBar position="static" elevation={0}>
          <Toolbar variant="dense">
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Behavysis Viewer
            </Typography>
            <Button color="inherit" onClick={handleOpenConfig}>
              Open
            </Button>
            <Button color="inherit" onClick={handleSave}>
              Save
            </Button>
          </Toolbar>
        </AppBar>

        <Box sx={{ flexGrow: 1, p: 2, overflow: "hidden" }}>
          <Grid container spacing={2} sx={{ height: "100%" }}>
            <Grid
              size={{ xs: 12, md: 8 }}
              sx={{ display: "flex", flexDirection: "column", gap: 2 }}
            >
              <VideoPlayer
                src={videoSrc}
                currentTime={currentTime}
                onTimeUpdate={setCurrentTime}
                onDurationChange={setDuration}
                keypointsModel={keypointsModel}
                fps={fps}
                showKeypoints={showKeypoints}
              />
              <TimelineGraph
                bouts={boutsData?.bouts || []}
                currentTime={currentTime}
                onSeek={setCurrentTime}
                fps={fps}
                windowSizeSeconds={5}
              />
            </Grid>
            <Grid
              size={{ xs: 12, md: 4 }}
              sx={{
                display: "flex",
                flexDirection: "column",
                gap: 2,
                height: "100%",
              }}
            >
              <Box sx={{ height: "60%" }}>
                <BoutsList
                  bouts={boutsData?.bouts || []}
                  selectedIndex={selectedBoutIndex}
                  onSelect={handleSelectBout}
                  fps={fps}
                />
              </Box>
              <Box sx={{ height: "40%" }}>
                <BoutInspector
                  bout={
                    boutsData && selectedBoutIndex >= 0
                      ? boutsData.bouts[selectedBoutIndex]
                      : null
                  }
                  index={selectedBoutIndex}
                  onUpdate={updateBout}
                  onReplay={() => {
                    if (boutsData && selectedBoutIndex >= 0) {
                      setCurrentTime(
                        boutsData.bouts[selectedBoutIndex].start / fps,
                      );
                    }
                  }}
                />
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default App;
