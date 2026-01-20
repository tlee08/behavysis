import { Table } from 'apache-arrow';

export interface KeypointConfig {
  colour_level: string;
  pcutoff: number;
  radius: number;
  cmap: string;
}

export interface IndivBpt {
  indiv: string;
  bpt: string;
}

export class KeypointsModel {
  table: Table | null = null;
  indivsBpts: IndivBpt[] = [];
  config: KeypointConfig = {
    colour_level: 'individuals',
    pcutoff: 0.9,
    radius: 5,
    cmap: 'viridis',
  };
  colours: string[] = [];

  constructor() {}

  load(table: Table, config: KeypointConfig) {
    this.table = table;
    this.config = config;
    this.parseIndivsBpts();
    this.generateColours();
  }

  private parseIndivsBpts() {
    if (!this.table) return;
    const fields = this.table.schema.fields.map(f => f.name);
    const uniquePairs = new Set<string>();
    fields.forEach(field => {
      const parts = field.split('_');
      if (parts.length >= 3) {
        uniquePairs.add(`${parts[0]}_${parts[1]}`);
      }
    });
    this.indivsBpts = Array.from(uniquePairs).map(pair => {
      const [indiv, bpt] = pair.split('_');
      return { indiv, bpt };
    });
  }

  private generateColours() {
    this.colours = this.indivsBpts.map((_, i) => {
      const hue = (i * 137.5) % 360;
      return `hsl(${hue}, 70%, 50%)`;
    });
  }

  drawKeypoints(ctx: CanvasRenderingContext2D, frameNum: number, width: number, height: number, originalWidth: number, originalHeight: number) {
    if (!this.table || frameNum < 0 || frameNum >= this.table.numRows) return;

    const row = this.table.get(frameNum);
    if (!row) return;

    const scaleX = width / originalWidth;
    const scaleY = height / originalHeight;

    this.indivsBpts.forEach((pair, i) => {
      const x = row[`${pair.indiv}_${pair.bpt}_x` as any];
      const y = row[`${pair.indiv}_${pair.bpt}_y` as any];
      const likelihood = row[`${pair.indiv}_${pair.bpt}_likelihood` as any];

      if (likelihood >= this.config.pcutoff) {
        ctx.beginPath();
        ctx.arc(x * scaleX, y * scaleY, this.config.radius, 0, 2 * Math.PI);
        ctx.fillStyle = this.colours[i];
        ctx.fill();
      }
    });
  }
}
