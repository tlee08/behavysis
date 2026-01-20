import React, { useRef, useEffect, useState } from 'react';
import { Box, IconButton, Slider, Typography, Stack } from '@mui/material';
import { PlayArrow, Pause } from '@mui/icons-material';
import { KeypointsModel } from '../models/KeypointsModel';

interface VideoPlayerProps {
  src: string | null;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  onDurationChange: (duration: number) => void;
  keypointsModel: KeypointsModel | null;
  fps: number;
  showKeypoints: boolean;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  currentTime,
  onTimeUpdate,
  onDurationChange,
  keypointsModel,
  fps,
  showKeypoints,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (videoRef.current && Math.abs(videoRef.current.currentTime - currentTime) > 0.1) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (playing) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setPlaying(!playing);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      onTimeUpdate(videoRef.current.currentTime);
      drawOverlay();
    }
  };

  const drawOverlay = () => {
    if (canvasRef.current && videoRef.current && keypointsModel && showKeypoints) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        const frameNum = Math.floor(videoRef.current.currentTime * fps);
        keypointsModel.drawKeypoints(
          ctx,
          frameNum,
          canvasRef.current.width,
          canvasRef.current.height,
          videoRef.current.videoWidth,
          videoRef.current.videoHeight
        );
      }
    } else if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  useEffect(() => {
      const resizeCanvas = () => {
          if (canvasRef.current && videoRef.current) {
              canvasRef.current.width = videoRef.current.clientWidth;
              canvasRef.current.height = videoRef.current.clientHeight;
              drawOverlay();
          }
      };
      window.addEventListener('resize', resizeCanvas);
      return () => window.removeEventListener('resize', resizeCanvas);
  }, [keypointsModel, showKeypoints]);

  return (
    <Box sx={{ position: 'relative', width: '100%', bgcolor: 'black', borderRadius: 1, overflow: 'hidden' }}>
      {src && (
        <>
          <video
            ref={videoRef}
            src={src}
            style={{ width: '100%', display: 'block' }}
            onTimeUpdate={handleTimeUpdate}
            onDurationChange={(e) => onDurationChange(e.currentTarget.duration)}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none',
              width: '100%',
              height: '100%',
            }}
          />
        </>
      )}
      {!src && (
        <Box sx={{ height: 360, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography color="grey.500">No video loaded</Typography>
        </Box>
      )}

      <Box sx={{ p: 1, bgcolor: 'rgba(0,0,0,0.7)', position: 'absolute', bottom: 0, left: 0, right: 0 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <IconButton size="small" onClick={togglePlay} sx={{ color: 'white' }}>
            {playing ? <Pause /> : <PlayArrow />}
          </IconButton>
          <Typography variant="caption" sx={{ color: 'white', minWidth: 100 }}>
            {currentTime.toFixed(2)}s / {(videoRef.current?.duration || 0).toFixed(2)}s
          </Typography>
          <Slider
            size="small"
            value={currentTime}
            max={videoRef.current?.duration || 0}
            step={0.01}
            onChange={(_, value) => {
              if (videoRef.current) videoRef.current.currentTime = value as number;
            }}
            sx={{ color: 'primary.main' }}
          />
        </Stack>
      </Box>
    </Box>
  );
};

export default VideoPlayer;
