import React, { useMemo } from 'react';
import Chart from 'react-apexcharts';
import { Box } from '@mui/material';
import { Bout } from '../models/Bout';

interface TimelineGraphProps {
  bouts: Bout[];
  currentTime: number;
  onSeek: (time: number) => void;
  fps: number;
  windowSizeSeconds: number;
}

const TimelineGraph: React.FC<TimelineGraphProps> = ({
  bouts,
  currentTime,
  onSeek,
  fps,
  windowSizeSeconds,
}) => {
  const VALUE2COLOR = {
    '-1': '#BDBDBD',
    '0': '#FF5252',
    '1': '#69F0AE',
  };

  const series = useMemo(() => {
    const behaviors = Array.from(new Set(bouts.map(b => b.behav)));
    return behaviors.map(behav => {
      const behavBouts = bouts.filter(b => b.behav === behav);
      return {
        name: behav,
        data: behavBouts.map(b => ({
          x: behav,
          y: [b.start / fps, b.stop / fps],
          fillColor: VALUE2COLOR[b.actual.toString() as keyof typeof VALUE2COLOR]
        }))
      };
    });
  }, [bouts, fps]);

  const options: ApexCharts.ApexOptions = {
    chart: {
      type: 'rangeBar',
      height: 200,
      events: {
        click: (_event, _chartContext, config) => {
          const { seriesIndex, dataPointIndex } = config;
          if (seriesIndex > -1 && dataPointIndex > -1) {
              const bout = bouts.filter(b => b.behav === series[seriesIndex].name)[dataPointIndex];
              onSeek(bout.start / fps);
          }
        }
      },
      toolbar: { show: false },
      animations: { enabled: false }
    },
    plotOptions: {
      bar: {
        horizontal: true,
        barHeight: '70%',
        rangeBarGroupRows: true
      }
    },
    xaxis: {
      type: 'numeric',
      labels: {
        formatter: (val) => `${Number(val).toFixed(1)}s`
      },
      min: Math.max(0, currentTime - windowSizeSeconds),
      max: currentTime + windowSizeSeconds,
    },
    annotations: {
      xaxis: [
        {
          x: currentTime,
          borderColor: '#FEB019',
          label: {
            style: { color: '#fff', background: '#FEB019' },
            text: 'Now'
          }
        }
      ]
    }
  };

  return (
    <Box sx={{ bgcolor: 'white', borderRadius: 1, p: 1, height: 250 }}>
      <Chart
        options={options}
        series={series}
        type="rangeBar"
        height="100%"
      />
    </Box>
  );
};

export default TimelineGraph;
