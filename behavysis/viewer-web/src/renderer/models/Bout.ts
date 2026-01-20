export interface Bout {
  start: number;
  stop: number;
  dur: number;
  behav: string;
  actual: number; // -1: Undetermined, 0: Non-Behav, 1: Behav
  user_defined: Record<string, number>;
}

export interface BoutStruct {
  behav: string;
  user_defined: string[];
}

export interface BoutsData {
  start: number;
  stop: number;
  bouts: Bout[];
  bouts_struct: BoutStruct[];
}
