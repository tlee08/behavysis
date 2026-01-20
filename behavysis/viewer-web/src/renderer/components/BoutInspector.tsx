import React from 'react';
import {
  Box,
  Typography,
  Radio,
  RadioGroup,
  FormControlLabel,
  FormControl,
  FormLabel,
  Checkbox,
  FormGroup,
  Divider,
  Paper,
  Button,
} from '@mui/material';
import { Bout } from '../models/Bout';

interface BoutInspectorProps {
  bout: Bout | null;
  index: number;
  onUpdate: (updates: Partial<Bout>) => void;
  onReplay: () => void;
}

const BoutInspector: React.FC<BoutInspectorProps> = ({ bout, index, onUpdate, onReplay }) => {
  if (!bout) {
    return (
      <Paper sx={{ p: 2, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <Typography color="grey.500">Select a bout to inspect</Typography>
      </Paper>
    );
  }

  const handleActualChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onUpdate({ actual: parseInt(event.target.value) });
  };

  const handleUserDefinedChange = (key: string, checked: boolean) => {
    const newUserDefined = { ...bout.user_defined, [key]: checked ? 1 : 0 };
    onUpdate({ user_defined: newUserDefined });
  };

  return (
    <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Inspector: {bout.behav} - {index}
      </Typography>
      <Divider sx={{ mb: 2 }} />

      <Button variant="outlined" fullWidth onClick={onReplay} sx={{ mb: 2 }}>
        Replay Bout
      </Button>

      <FormControl component="fieldset" sx={{ mb: 2 }}>
        <FormLabel component="legend">Actual Behavior?</FormLabel>
        <RadioGroup value={bout.actual.toString()} onChange={handleActualChange}>
          <FormControlLabel value="1" control={<Radio size="small" />} label="Is Behavior" />
          <FormControlLabel value="0" control={<Radio size="small" />} label="Not Behavior" />
          <FormControlLabel value="-1" control={<Radio size="small" />} label="Undetermined" />
        </RadioGroup>
      </FormControl>

      <Divider sx={{ mb: 2 }} />

      <Typography variant="subtitle2" gutterBottom>
        User Defined Attributes
      </Typography>
      <FormGroup>
        {Object.entries(bout.user_defined).map(([key, value]) => (
          <FormControlLabel
            key={key}
            control={
              <Checkbox
                size="small"
                checked={value === 1}
                onChange={(e) => handleUserDefinedChange(key, e.target.checked)}
              />
            }
            label={key}
          />
        ))}
      </FormGroup>
    </Paper>
  );
};

export default BoutInspector;
