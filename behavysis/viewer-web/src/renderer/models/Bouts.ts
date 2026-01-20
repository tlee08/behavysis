import { Bout, BoutsData, BoutStruct } from './Bout';

export class BoutsModel {
  data: BoutsData;

  constructor(data?: BoutsData) {
    this.data = data || {
      start: 0,
      stop: 0,
      bouts: [],
      bouts_struct: [],
    };
  }

  get bouts() {
    return this.data.bouts;
  }

  set bouts(val: Bout[]) {
    this.data.bouts = val;
  }

  get bouts_struct() {
    return this.data.bouts_struct;
  }

  updateBout(index: number, updates: Partial<Bout>) {
    if (index >= 0 && index < this.data.bouts.length) {
      this.data.bouts[index] = { ...this.data.bouts[index], ...updates };
    }
  }
}
