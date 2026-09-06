/* ============================================================================
 * aa_features.h — on-chip port of the remaining arterialage signal-processing.
 *
 * Companion to arterialage.h (which holds the dtau core). This file ports:
 *   activity.py     -> ENMO, epochs, bouts                (aa_enmo, aa_epochs_*, aa_bouts_*)
 *   features/rr_irregularity.py -> RR-irregularity burden (aa_rr_irregularity)
 *   features/atc95.py           -> ATC95 percentile       (aa_atc95)
 *   features/hrr60.py           -> HR recovery @60s        (aa_hrr60)
 *   features/hracs.py           -> HR-vs-activity slope    (aa_hracs)
 *   features/dtau_derived.py    -> dipping, PWV reserve    (aa_dipping, aa_reserve)
 *
 * MATH UNCHANGED from the validated Python. Only language + fixed buffers
 * differ. Percentile / median use the same "linear interpolation" definition
 * numpy uses, so results match.
 * ========================================================================== */
#ifndef AA_FEATURES_H
#define AA_FEATURES_H

#include <stdint.h>
#include <stdbool.h>

#ifndef AA_MAX_EPOCHS
#define AA_MAX_EPOCHS 4096     /* activity epochs per record            */
#endif
#ifndef AA_MAX_BOUTS
#define AA_MAX_BOUTS  256      /* bouts per record                      */
#endif
#ifndef AA_MAX_RR
#define AA_MAX_RR     2048     /* RR intervals per record               */
#endif
#ifndef AA_MAX_HR_PTS
#define AA_MAX_HR_PTS 512      /* (intensity,HR) points for HRACS       */
#endif

/* ---- frozen constants (verbatim from constants.py) ---------------------- */
#define AA_EPOCH_S            15.0f
#define AA_BOUT_THRESHOLD_G   0.100f
#define AA_BOUT_MIN_S         60.0f
#define AA_BOUT_GAP_TOL_S     30.0f

/* autocalibration (activity.py) */
#define AA_CAL_WINDOW_S        10.0f
#define AA_CAL_STILL_SD_G      0.013f
#define AA_CAL_SPHERE_MIN_G    0.3f
#define AA_CAL_MIN_STILL_POINTS 50
#define AA_CAL_MAX_ITER        25
#ifndef AA_CAL_MAX_STILL
#define AA_CAL_MAX_STILL       4096   /* max still-window points held for the fit */
#endif

#define AA_ATC_PERCENTILE     95.0f
#define AA_ATC_MIN_BOUTS      10

#define AA_RR_WINDOW_BEATS    60
#define AA_RR_DEV_FRAC        0.25f
#define AA_RR_MIN_S           0.20f
#define AA_RR_MAX_S           2.5f
#define AA_RR_MIN_INTERVALS   30

#define AA_HRR_RECOVERY_S     60.0f
#define AA_HRR_HR_WINDOW_S    10.0f
#define AA_HRR_MIN_BOUT_S     60.0f
#define AA_HRR_SETTLE_TOL_BPM 8.0f
#define AA_HRR_MIN_PEAK_BPM   90.0f
#define AA_HRR_MAX_GAP_S      5.0f

#define AA_HRACS_WINDOW_S            30.0f
#define AA_HRACS_MIN_POINTS         5
#define AA_HRACS_MIN_INTENSITY_RANGE_G 0.03f
#define AA_HRACS_MIN_R2             0.30f

#define AA_DIP_MIN_BEATS_PER_PHASE     30
#define AA_RESERVE_MIN_BEATS_PER_PHASE 30

/* generic status for the value-or-refusal features */
typedef enum { AA_FEAT_OK = 0, AA_FEAT_REFUSED = 1 } aa_feat_status_t;

/* =================== activity chain (activity.py) ======================== */

/* accelerometer calibration: per-axis offset and gain so |a|==1g when still */
typedef struct {
    float offset[3];
    float scale[3];
    bool  converged;      /* false => use raw axes; calibration refused        */
    int   n_still;        /* number of still-window points used                */
    float residual_mg;    /* mean |‖corrected‖ - 1g| over still points, in mg  */
} aa_calib_t;

/* Fit calibration from an accel record (n_samples*3 interleaved xyz, in g).
 * Mirrors autocalibrate(): still-window detection, sphere-coverage refusal,
 * alternating least squares. On refusal returns converged=false (identity cal). */
aa_calib_t aa_autocalibrate(const float *accel_xyz, int n_samples, float fs);

/* ENMO: euclidean norm of one accel sample minus 1g, clipped at 0 (g).
 * Pass calibrated axes (offset/scale already applied) or raw if uncalibrated. */
float aa_enmo_sample(float ax, float ay, float az);
/* ENMO with a calibration applied (offset/scale) — matches enmo(a, cal). */
float aa_enmo_sample_cal(float ax, float ay, float az, const aa_calib_t *cal);

/* mean ENMO per fixed epoch. accel is n*3 interleaved (ax,ay,az,...).
 * Writes epoch mean ENMO into epoch_enmo[], epoch start times into epoch_t0[]
 * (t0 = i*epoch_s), returns number of epochs. */
int aa_epochs_from_accel(const float *accel_xyz, int n_samples, float fs,
                         float *epoch_enmo, float *epoch_t0, int max_epochs);

/* one bout = start/end time (s) + mean ENMO (g) */
typedef struct { float t_start, t_end, mean_enmo_g; int n_epochs; } aa_bout_t;

/* group above-threshold epochs into bouts (tolerating brief dips).
 * epoch_t0[] are epoch start times, epoch_dt = epoch duration (s).
 * Writes bouts[], returns bout count. */
int aa_bouts_from_epochs(const float *epoch_enmo, const float *epoch_t0,
                         int n_epochs, float epoch_dt,
                         aa_bout_t *bouts, int max_bouts);

/* =================== Feature 1: RR irregularity ========================== */
/* rr_dur_s[] = interval durations (s); rr_gap[] = per-interval spans_gap flag.
 * Returns burden (fraction) via *burden; status refused if too few valid. */
aa_feat_status_t aa_rr_irregularity(const float *rr_dur_s, const bool *rr_gap,
                                    int n, float *burden, int *n_valid);

/* =================== Feature 2: ATC95 ==================================== */
/* bout_mean_enmo[] = per-bout mean ENMO (g). Returns 95th percentile. */
aa_feat_status_t aa_atc95(const float *bout_mean_enmo, int n_bouts,
                          float *value);

/* =================== Feature 6: HRR60 =================================== */
/* windowed HR (bpm) at t_centre over beat_times[] (sorted, seconds).
 * Returns true and sets *hr on success. */
bool aa_hr_at(const float *beat_times, int n_beats, float t_centre,
              float window_s, float max_gap_s, float *hr);

/* HRR60 across bouts; reports MAX drop (bpm). beat_times sorted. */
aa_feat_status_t aa_hrr60(const float *beat_times, int n_beats,
                          const aa_bout_t *bouts, int n_bouts,
                          float *hrr_bpm);

/* =================== Feature 5: HRACS =================================== */
/* slope of HR (bpm) on intensity (g) via least squares. */
aa_feat_status_t aa_hracs_slope(const float *intensity, const float *hr,
                                int n, float *slope, float *r2);

/* end-to-end HRACS: windows the record, forms (ENMO, HR) points, fits slope. */
aa_feat_status_t aa_hracs(const float *beat_times, int n_beats,
                          const float *enmo_series, int enmo_n, float enmo_fs,
                          float t_start, float t_end, float *slope, float *r2);

/* =================== Features 3 & 4: dipping / reserve =================== */
/* dipping = (dtau_day - dtau_night)/|dtau_day|. n_*: beat support per phase. */
aa_feat_status_t aa_dipping(float dtau_day_ms, int n_day,
                            float dtau_night_ms, int n_night, float *frac);
/* reserve = dtau_rest - dtau_active (ms). */
aa_feat_status_t aa_reserve(float dtau_rest_ms, int n_rest,
                            float dtau_active_ms, int n_active, float *delta_ms);

/* shared helpers (numpy-compatible) */
float aa_median(float *a, int n);                 /* sorts a in place */
float aa_percentile(float *a, int n, float p);    /* linear interp, sorts a */

#endif /* AA_FEATURES_H */