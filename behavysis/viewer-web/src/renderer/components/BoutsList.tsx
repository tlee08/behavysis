import React from 'react';
import { List, ListItem, ListItemText, ListItemButton, Paper, Typography, Box } from '@mui/material';
import { Bout } from '../models/Bout';

interface BoutsListProps {
  bouts: Bout[];
  selectedIndex: number;
  onSelect: (index: number) => void;
  fps: number;
}

const BoutsList: React.FC<BoutsListProps> = ({ bouts, selectedIndex, onSelect, fps }) => {
  const VALUE2COLOR = {
    '-1': '#BDBDBD',
    '0': '#FF5252',
    '1': '#69F0AE',
  };

  return (
    <Paper sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ p: 1, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Bouts</Typography>
      </Box>
      <List sx={{ flexGrow: 1, overflow: 'auto' }}>
        {bouts.map((bout, index) => (
          <ListItem key={index} disablePadding>
            <ListItemButton
              selected={selectedIndex === index}
              onClick={() => onSelect(index)}
              sx={{
                borderLeft: 6,
                borderColor: VALUE2COLOR[bout.actual.toString() as keyof typeof VALUE2COLOR],
              }}
            >
              <ListItemText
                primary={`${bout.behav} (${index})`}
                secondary={`${(bout.start / fps).toFixed(2)}s - ${(bout.stop / fps).toFixed(2)}s`}
              />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default BoutsList;
