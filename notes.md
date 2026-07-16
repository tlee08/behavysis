Here's my analysis:

---

## Q3: Is the `_compute_deviations` merge correct?

**Yes.** The single-pass merge is clean and correct. The `or "_mean_" in key` branch replaces the old second loop. The `"_deviation"` and `"_percentile_rank"` skips are handled once at the top.

One thing I notice: `_compute_deviations` still includes `sum_probabilities` in its `_prefixes` (line 571), which is correct — deviating tracking quality makes sense ("is this frame worse than average?"). And `_compute_rolling` no longer has `sum_probabilities` — that's correct since you removed it.

There's a slight inconsistency though: `_compute_deviations` has `_suffixes` that don't include `_cdist_sum` or `_hull_area_change`, while `_compute_rolling` does include them. Should `_hull_area_change` and individual `_cdist_sum` get deviations too? SimBA deviates hull area and cdist stats. Worth aligning for consistency, but not a bug — function dependent.

---

## Q2: The claim about `cdist_sum_` prefix — is it correct?

**The claim is wrong.** Let me prove it.

The feature name is `"cdist_sum_all"`. Your prefix is `"cdist_sum_"`.

```python
"cdist_sum_all".startswith("cdist_sum_")   # True
"total_movement_all".startswith("total_movement_")  # True
```

Both grand totals ARE matched and WILL be rolled. The trailing underscore in the prefix does NOT prevent matching `_all` — `"cdist_sum_"` literally IS a prefix of `"cdist_sum_all"` (the string `cdist_sum_` is the first 10 characters of `cdist_sum_all`).

You can verify in Python:

```python
>>> "cdist_sum_all".startswith("cdist_sum_")
True
>>> "total_movement_all".startswith("total_movement_")
True
```

The person who made this claim likely assumed `startswith` does full-token matching or got confused by the underscore placement. Both grand totals are correctly included in rolling.

---

## Q3: Do we calculate all the features SimBA does?

**No, but close — ~85-90% coverage.** Here's what we're missing:

### SimBA features we DON'T compute:

| Feature family                         | SimBA examples                           | Why we miss them                                                                                                                                    |
| -------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **3-point angles**                     | `Mouse_1_angle`, `Total_angle_both_mice` | Requires knowing which 3 bodyparts form a meaningful angle (Nose-Center-Tail_base). Can't generalize without semantics.                             |
| **Path tortuosity**                    | `Tortuosity_Mouse1_{window}`             | Path-winding metric. SimBA uses `as_strided` trick over centroid trajectory. Computationally heavy, not trivial to generalize.                      |
| **Centroid-specific movement rolling** | `Centroid_movement_M1_median_{w}`        | SimBA rolls individual centroid/tail/nose movements. We deliberately excluded per-bodypart movement from rolling (by design in your new allowlist). |

### SimBA features we COMPUTE differently (roughly equivalent):

| SimBA concept                                                         | Our equivalent                                              |
| --------------------------------------------------------------------- | ----------------------------------------------------------- |
| Semantic distances (nose-to-tail, ear distance, width) → 24 hardcoded | Combinatorial within/cross pairwise distances → 120 generic |
| 3 aggregate movement totals (centroids, tail_ends, all_bodyparts)     | Per-individual `total_movement_*` + `total_movement_all`    |
| `M1_hull_large/small/mean/sum_euclidean`                              | `{indiv}_cdist_max/min/mean/sum`                            |
| `Sum_euclidean_distance_hull_M1_M2`                                   | `cdist_sum_all`                                             |

### What we compute that Simba DOESN'T:

- **120 combinatorial distances** vs SimBA's 24. SimBA has semantic coverage of known-important pairs; we have exhaustive coverage. A classifier can discover non-obvious signal.
- **Generic per-bodypart movement** for EVERY bodypart. Same in number (8 bp × 2 animals = 16), but ours are named generically.

### The real gap: angles and tortuosity

These are the two feature families that are truly missing and might matter for behavioral classification:

- **Angles**: A 3-point body angle (Nose-Center-Tail_base) captures posture/orientation. Adding a generic angle computation (all 3-point combinations?) would be combinatorial explosion. Adding it with semantic knowledge (e.g., a config that names which 3 bodyparts form an angle) would require design.
- **Tortuosity**: Path complexity over time. This is window-based and could be computed generically from centroid movement. Worth considering as a future feature group.

Both require semantic bodypart roles, which goes against the generic design. If you want to add them, they should be opt-in via `ExtractFeaturesConfig` (e.g., `angle_triplets: list[list[str]]`).

---

### Summary

| Aspect                                               | Verdict                                                                                         |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Q3: `_compute_deviations` merge                      | Correct                                                                                         |
| Q3: `_compute_rolling` prefix logic for grand totals | **Wrong claim** — both `total_movement_all` and `cdist_sum_all` are matched correctly           |
| SimBA feature coverage                               | ~85-90%. Missing angles and tortuosity (both semantic). Gaining richer combinatorial distances. |
