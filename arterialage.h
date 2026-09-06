/* ============================================================================
 * arterialage.h  —  on-chip signal-processing port of the arterialage pipeline
 *
 * This is a faithful C translation of the validated Python reference
 * (arterialage/*.py). The MATH IS UNCHANGED — every constant and every
 * operation mirrors the Python, which is validated on datasets. Only the
 * language changed (Python -> C), plus the mechanical things C forces:
 * fixed-size buffers instead of growing lists, explicit loops instead of numpy.
 *
 * Pipeline (per capture, both ears already sampled on one exposure clock):
 *    raw green (+ red/ir) per ear
 *      -> bandpass  (frozen Butterworth SOS, dsp.py)            [timing signal]
 *      -> adaptive-threshold peak detect (streaming.py)         [beat anchors]
 *      -> quality gate  PI + red/IR ratio (quality.py)          [drop bad beats]
 *      -> foot via intersecting tangents (dtau_ref.py L1)       [timing fiducial]
 *      -> pair feet across ears (dtau_ref.py L2)
 *      -> robust dtau: hard-reject / MAD / median (dtau_ref.py L3)
 *      => dtau in milliseconds (positive => pulse reaches LEFT ear first)
 *
 * Each "Streaming*" struct here maps 1:1 onto the Python class of the same name.
 * ========================================================================== */
#ifndef ARTERIALAGE_H
#define ARTERIALAGE_H

#include <stdint.h>
#include <stdbool.h>

/* ---- capacities (compile-time fixed; C has no growing lists) ------------- */
#ifndef AA_MAX_SAMPLES
#define AA_MAX_SAMPLES 4096   /* max PPG samples per capture window per ear   */
#endif
#ifndef AA_MAX_BEATS
#define AA_MAX_BEATS   512    /* max detected beats per ear per capture       */
#endif
#define AA_SOS_SECTIONS 4     /* 4th-order band-pass = 4 biquad sections      */
#define AA_SEARCH_MAX  128    /* search_buf cap; SEARCH_S*fs = 0.10*1024 ≈ 103 */

/* ---- frozen constants (verbatim from constants.py) ---------------------- */
#define AA_FS_HZ            1024.0f
#define AA_THR_FRAC_RHYTHM  0.45f
#define AA_REFRACTORY_S     0.20f
#define AA_SEARCH_S         0.10f
#define AA_ENV_TAU_S        4.0f
#define AA_ENV_RISE         0.15f
#define AA_SEED_S           1.0f

#define AA_DTAU_FOOT_UPSTROKE_FRAC 0.30f
#define AA_DTAU_BASELINE_MS        40.0f
#define AA_DTAU_PAIR_WINDOW_MS     100.0f
#define AA_DTAU_MAX_ABS_MS         50.0f
#define AA_DTAU_MAD_K              5.0f
#define AA_DTAU_MIN_BEATS          30
#define AA_DTAU_MIN_HR_BPM         30.0f
#define AA_DTAU_MAX_HR_BPM         220.0f

#define AA_PI_MIN_FRAC     0.001f
#define AA_REDIR_RATIO_LO  0.4f
#define AA_REDIR_RATIO_HI  3.0f
#define AA_MOTION_ENMO_G   0.05f

/* ===== streaming biquad (streaming.py StreamingBiquad) ==================== */
typedef struct {
    float  b0[AA_SOS_SECTIONS], b1[AA_SOS_SECTIONS], b2[AA_SOS_SECTIONS];
    float  a1[AA_SOS_SECTIONS], a2[AA_SOS_SECTIONS];
    double z1[AA_SOS_SECTIONS], z2[AA_SOS_SECTIONS];   /* state in double: IIR
                                     feedback drifts badly in float32 over time */
} aa_biquad_t;

void  aa_biquad_init(aa_biquad_t *bq);     /* loads frozen coeffs, zeros state */
void  aa_biquad_reset(aa_biquad_t *bq);    /* zeros state only                 */
float aa_biquad_push(aa_biquad_t *bq, float x);

/* ===== streaming peak detector (streaming.py StreamingPeakDetector) ======= */
typedef enum { AA_IDLE = 0, AA_SEARCH = 1, AA_SKIP = 2 } aa_pd_mode_t;

typedef struct {
    float   fs, thr_frac, env_decay, env_rise;
    int32_t refr, search;                 /* in samples */
    float   env;
    int32_t last_idx, n;
    aa_pd_mode_t mode;
    float   search_buf[AA_SEARCH_MAX];
    int32_t search_i, cross_idx, skip_until;
    float   prev_x;
    uint32_t seq;
    bool    seeded;
    int32_t seed_n;
    float   seed_max;
    /* output of the most recent confirmed beat */
    int32_t last_beat_idx;                /* sample index of confirmed peak    */
    float   last_beat_amp;
} aa_peakdet_t;

void aa_peakdet_init(aa_peakdet_t *d, float fs);
void aa_peakdet_reset(aa_peakdet_t *d);
/* returns true and sets *beat_idx when a beat is confirmed on this sample */
bool aa_peakdet_push(aa_peakdet_t *d, float x, int32_t *beat_idx);
/* end-of-stream: emit a pending partially-searched beat if any */
bool aa_peakdet_flush(aa_peakdet_t *d, int32_t *beat_idx);

/* ===== high-level: run detector over a whole buffer ====================== */
/* Bandpass `raw` into `ac` (both length n), then detect peaks; fills
 * peak_idx[] up to max_peaks, returns count. Mirrors adaptive_threshold_peaks. */
int aa_detect_peaks(const float *raw, int n, float fs,
                    float *ac_out, int32_t *peak_idx, int max_peaks);

/* ===== quality gate (quality.py) ========================================= */
/* perfusion index AC/DC over the beat window on RAW counts */
float aa_perfusion_index(const float *raw, int n, float fs, int peak_idx);
/* gate one beat; returns true if it passes. red/ir may be NULL (skip ratio). */
bool aa_gate_beat(const float *timing_raw, int n, float fs, int peak_idx,
                  const float *red_raw, const float *ir_raw);

/* ===== dtau_ref (features/dtau_ref.py) =================================== */
/* L1: foot time (seconds) for each peak via intersecting tangents.
 * Writes feet_s[] (len = number of valid feet), returns that count. */
int aa_foot_tangent(const float *sig, int n, float fs,
                    const int32_t *peak_idx, int n_peaks, float *feet_s);

/* L2+L3: pair feet across ears and reduce to one robust dtau in ms.
 * Returns true and sets *dtau_ms on success; false + reason code on refusal. */
typedef enum {
    AA_DTAU_OK = 0,
    AA_DTAU_NO_PAIRS,
    AA_DTAU_ALL_BEYOND_CUTOFF,
    AA_DTAU_TOO_FEW_BEATS,
    AA_DTAU_HR_IMPLAUSIBLE
} aa_dtau_status_t;

aa_dtau_status_t aa_dtau_from_feet(const float *feet_l, int nL,
                                   const float *feet_r, int nR,
                                   float *dtau_ms, int *n_final);

/* ===== end-to-end convenience =========================================== */
/* Given both ears' RAW green buffers, produce dtau in ms.
 * scratch_ac_l / scratch_ac_r must each be >= n floats (bandpass output). */
aa_dtau_status_t aa_dtau_from_green(const float *green_l_raw,
                                    const float *green_r_raw, int n, float fs,
                                    float *ac_l, float *ac_r,
                                    float *dtau_ms, int *n_final);

#endif /* ARTERIALAGE_H */
