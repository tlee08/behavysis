I have a large dataset 500,000 examples with 200 features and a binary label. The dataset is a concatenated set of timeseries (i.e. data + labels corresponding to a video).
I have trained a xgb model and TabPFN model. The training set is a set of videos and test set is another set of videos, split stratified by the label 80/20 split. Labels are unbalanced (about 5% examples are label 1, 95% are label 0).
The XGB model has medium training PR and ROC curve performance a lower test performance. The TabPFN has almost perfect PR and ROC curve performance and test performance similar to the xgb model.
Your job: postulate why this is happening. Give first-principles data science justification for why we are seeing this.

Specifically:

- why is the performance for test lower? Are we overfitting, or simply not getting the features that capture signal for the labels?
- TabPFN training set is basically perfect and then test set is unremarkable. Is this overfitting? What can we do about it?
- how can I investigate where this pipeline has limitations and so I can begin to improve them? My thoughts of possible limitations and points of improvement are a) keypoints extraction model is not good enough, b) we’re not deriving features from the keypoints that capture good enough signal of the label (behaviour), c) even though we have many rows in our dataset, these are timeseries so adjacent frames are very similar… we maybe don’t actually have enough different kinds of semantic “examples” of the behaviour.

Be a data scientist. Think critically, then provide your assessment succinctly.

---

---

**Short answer: this is not "just overfitting" in either model — it looks like an _effective sample size / label-leakage-via-autocorrelation_ problem, plus possibly video-identity shortcuts. Both models are hitting the same test ceiling, which is the most informative clue.**

**1. Why is test lower than train (XGB)?**

Classic overfitting is one explanation, but the fact that XGB's test ceiling matches TabPFN's test ceiling (a completely different inductive bias — trees vs. in-context kernel method) is a strong signal that the _gap is data-driven, not model-driven_. If it were pure overfitting to noise, you'd expect two very different architectures to overfit differently and land at different test scores. Instead:

- Both models are bounded by the same information ceiling in the features/videos → suggests the **features don't carry enough video-invariant signal**, and/or there's **video-level distribution shift** (different subjects, camera angles, lighting, arena, baseline behavior rates) that the model can't bridge.
- Some genuine overfitting is likely also present in XGB (trees can memorize idiosyncratic frame patterns), but it's probably a secondary effect on top of the shift.

**2. Is TabPFN's near-perfect train / mediocre test "overfitting"? What to do?**

Be careful here — this is very likely **not real overfitting**, it's a **train-metric artifact from your timeseries structure**:

- TabPFN's mechanism is essentially in-context similarity/kernel-based inference — it classifies a query point based on how similar it is to context points. If you evaluate "training performance" by feeding the training set as context and then scoring those same rows, adjacent frames from the same behavioral bout are near-duplicates of each other. So the model is basically doing near-neighbor lookup against (almost) itself — this will trivially look "almost perfect" regardless of whether any generalizable signal exists.
- This isn't overfitting in the classical bias/variance sense; it's a **measurement problem**. The "train score" here is not a meaningful diagnostic at all for TabPFN — discard it.
- What to do: never trust train-set-as-context self-evaluation for TabPFN on autocorrelated data. Use **grouped cross-validation** (leave-one-video-out, or leave-N-videos-out) as your only real signal of generalization, for both models.

**3. Where's the pipeline actually limited? How to investigate.**

Your three hypotheses are the right lens. Concretely:

- **Effective sample size (your hypothesis c) — check this first.** 500k _rows_ is not 500k _independent examples_. Count the number of distinct positive-label _bouts/events_ (contiguous runs of label=1) in train and test, not frames. If you have, say, 25,000 positive frames but only 40 independent bouts, your real N for the positive class is ~40, and that's why train "looks" solved (interpolating within bouts) but test fails (new bouts/subjects don't interpolate). This is very likely the dominant issue.
- **Video-identity shortcut / distribution shift — run adversarial validation.** Train a classifier to distinguish train-video rows from test-video rows using your 200 features. If that's trivially easy (high AUC), your models may be partly learning "which video/subject this is" rather than the behavior itself — a shortcut that inflates apparent signal without generalizing.
- **Per-video test performance breakdown.** Don't look at aggregate PR/ROC only — plot AUPRC per individual test video. If a few videos drag the average down while most are fine, that points to subject/context-specific failure (keypoint quality or behavior-expression differences), not a global feature deficiency.
- **Keypoint extraction quality (a).** Check keypoint confidence/occlusion metrics and correlate with prediction errors; visually inspect false negatives/positives directly on the video frames near label transitions.
- **Feature engineering (b).** If features are per-frame/instantaneous, they likely underrepresent temporal dynamics (velocity, acceleration, rolling windows, dynamics of behavior onset/offset) — a real behavioral signature is usually in the _trajectory_, not a single frame. Frame-independent tabular models (XGB/TabPFN) discard this. Worth testing a sequence-aware model (1D CNN / temporal CNN / small transformer over windows) as a sanity check on whether temporal context substantially improves test AUPRC — that would confirm feature engineering, not sample diversity, is the bottleneck.
- **Sample diversity (c) fix.** More videos/subjects generalizes much better than more frames from existing videos. Prioritize collecting new sessions over increasing frame density.

**Practical next step:** run leave-one-video-out (or leave-5-videos-out repeated) CV for both models and plot the variance in AUPRC across folds. High variance across folds = you're data-limited (few independent bouts/subjects). Low variance but still a ceiling = feature/signal-limited. That single experiment will tell you which of your three hypotheses to prioritize.

---

---

I have concerns that the data volume will become too much after getting more videos (marginal gain for much more data, which takes more training time and resources).

How should I handle this? How should I sample this?

---

---

Good instinct — more data isn't free, and given the autocorrelation problem, **most of your marginal frames carry almost no new information**. The fix is to sample for _diversity_, not _volume_.

**1. Diagnose before deciding how much to collect — run a learning curve, but on independent units.**
Plot test AUPRC vs. number of _videos_ (or number of independent positive bouts) included in training, not number of frames. Train on 20%, 40%, 60%, 80% of your videos, holding the same video-level test set fixed. If the curve is still climbing steeply, more videos help a lot. If it's flattening, you're near saturation and should stop collecting and instead spend effort on features/architecture. This single plot answers your "is more data worth it" question directly, cheaply, using data you already have.

**2. Downsample the redundant majority — you almost certainly can train on far fewer frames without losing performance.**

- **Temporal decorrelation / stride sampling within bouts.** Instead of every frame, sample every Nth frame (or one frame per X ms) from within a behavior bout, and similarly subsample the background/negative class. Adjacent frames are near-duplicates — keeping all of them just makes optimization slower without adding signal, and inflates the illusion of "training performance."
- **Bout-based stratified sampling.** Sample a fixed number of frames _per bout_, not per video or globally, so long bouts don't dominate and short/rare behavior types aren't underrepresented.
- **Negative class subsampling / hard-negative mining.** With 95% negatives, you don't need all of them. Random-subsample easy negatives heavily, but retain (or oversample) negatives that are "near" positive-looking (high model uncertainty / near decision boundary from a first-pass model) — these are the informative ones for the boundary. This is standard for imbalanced classification and cuts volume a lot while preserving discriminative signal.
- **Deduplicate near-identical frames.** A simple embedding-distance or feature-distance threshold within a bout can flag and drop near-duplicate frames (frames where nothing meaningfully changed).

**3. When adding new videos, prioritize diversity over quantity.**

- Prefer many _different_ subjects/sessions/conditions with modest frames each, over few subjects with dense frames. Diversity of context (lighting, camera angle, individual variation) is what actually improves the video-level generalization gap you're diagnosing — raw frame count within already-seen contexts won't.
- If resources are limited, consider **active learning**: use your current model's per-video uncertainty/error to prioritize labeling additional videos where it fails, rather than blindly adding random new videos.

**4. Practical ceiling for your compute.**
Once you've picked a sampling scheme (e.g., stride-sampled + hard-negative-mined), you can likely cut total row count 5–20x with negligible loss versus using all raw frames — validate this empirically by comparing full-data vs. subsampled-data performance on your held-out video test set. If they're statistically indistinguishable, lock in the smaller sampled set as your standard training pipeline and reserve full-frame density only for a final validation pass, not iterative experimentation.

**Bottom line:** sample down for redundancy (stride/dedup within bouts, subsample easy negatives), sample up for diversity (more distinct videos/subjects, prioritized via error analysis or active learning), and let the video-level learning curve tell you when to stop collecting altogether.
