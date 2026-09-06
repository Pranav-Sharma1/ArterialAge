/* ============================================================================
 * aa_features.c — faithful C port of the remaining arterialage pipeline.
 * Math mirrors the validated Python; see aa_features.h.
 * ========================================================================== */
#include "aa_features.h"
#include <math.h>
#include <string.h>

/* ---- small sort + numpy-compatible median/percentile -------------------- */
static void aa__isort(float *a, int n)
{
    for (int i = 1; i < n; i++) {
        float k = a[i]; int j = i - 1;
        while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
        a[j + 1] = k;
    }
}

float aa_median(float *a, int n)
{
    if (n <= 0) return 0.0f;
    aa__isort(a, n);
    if (n & 1) return a[n / 2];
    return 0.5f * (a[n / 2 - 1] + a[n / 2]);
}

/* numpy.percentile default: linear interpolation on sorted data.
 * rank = p/100 * (n-1); interpolate between floor and ceil. */
float aa_percentile(float *a, int n, float p)
{
    if (n <= 0) return 0.0f;
    if (n == 1) return a[0];
    aa__isort(a, n);
    float rank = (p / 100.0f) * (float)(n - 1);
    int lo = (int)floorf(rank);
    int hi = (int)ceilf(rank);
    if (lo < 0) lo = 0;
    if (hi > n - 1) hi = n - 1;
    float frac = rank - (float)lo;
    return a[lo] + frac * (a[hi] - a[lo]);
}

/* ======================= activity.py ===================================== */

/* ---- autocalibrate: fit offset+scale so still points sit on unit sphere --- */
aa_calib_t aa_autocalibrate(const float *accel_xyz, int n_samples, float fs)
{
    aa_calib_t cal;
    cal.offset[0] = cal.offset[1] = cal.offset[2] = 0.0f;
    cal.scale[0]  = cal.scale[1]  = cal.scale[2]  = 1.0f;
    cal.converged = false; cal.n_still = 0; cal.residual_mg = 0.0f;

    int w = (int)lroundf(AA_CAL_WINDOW_S * fs); if (w < 1) w = 1;
    int n_win = n_samples / w;
    if (n_win == 0) return cal;

    /* still-window points on the sphere: per window, if all 3 axes have
     * population SD < still_sd_g, keep the window's per-axis mean. */
    static float pts[AA_CAL_MAX_STILL][3];
    int np_ = 0;
    for (int iw = 0; iw < n_win; iw++) {
        int base = iw * w;
        double mean[3] = {0,0,0};
        for (int k = 0; k < w; k++) {
            const float *s = &accel_xyz[(base + k) * 3];
            mean[0] += s[0]; mean[1] += s[1]; mean[2] += s[2];
        }
        mean[0] /= w; mean[1] /= w; mean[2] /= w;
        double var[3] = {0,0,0};
        for (int k = 0; k < w; k++) {
            const float *s = &accel_xyz[(base + k) * 3];
            for (int j = 0; j < 3; j++) { double d = s[j] - mean[j]; var[j] += d * d; }
        }
        bool still = true;
        for (int j = 0; j < 3; j++) {
            double sd = sqrt(var[j] / (double)w);   /* population SD, like numpy */
            if (!(sd < AA_CAL_STILL_SD_G)) { still = false; break; }
        }
        if (still && np_ < AA_CAL_MAX_STILL) {
            pts[np_][0] = (float)mean[0];
            pts[np_][1] = (float)mean[1];
            pts[np_][2] = (float)mean[2];
            np_++;
        }
    }
    cal.n_still = np_;
    if (np_ < AA_CAL_MIN_STILL_POINTS) return cal;   /* refuse: too few still */

    /* sphere coverage: each axis must span below -sphere_min and above +sphere_min */
    for (int j = 0; j < 3; j++) {
        float mn = pts[0][j], mx = pts[0][j];
        for (int i = 1; i < np_; i++) { if (pts[i][j] < mn) mn = pts[i][j]; if (pts[i][j] > mx) mx = pts[i][j]; }
        if (!(mn < -AA_CAL_SPHERE_MIN_G && mx > AA_CAL_SPHERE_MIN_G)) return cal; /* refuse */
    }

    /* alternating least squares: project each point onto unit sphere, regress
     * per axis (mean-centered slope/intercept), a few iterations. */
    double offset[3] = {0,0,0}, scale[3] = {1,1,1};
    for (int it = 0; it < AA_CAL_MAX_ITER; it++) {
        for (int j = 0; j < 3; j++) {
            /* build target unit vectors' j-th component + x = raw pts[:,j] */
            double xbar = 0.0, ybar = 0.0;
            static float tx[AA_CAL_MAX_STILL], ty[AA_CAL_MAX_STILL];
            for (int i = 0; i < np_; i++) {
                double cx = ((double)pts[i][0] - offset[0]) * scale[0];
                double cy = ((double)pts[i][1] - offset[1]) * scale[1];
                double cz = ((double)pts[i][2] - offset[2]) * scale[2];
                double nrm = sqrt(cx*cx + cy*cy + cz*cz); if (nrm == 0.0) nrm = 1.0;
                double comp = ((j==0)?cx:(j==1)?cy:cz) / nrm;   /* target unit comp */
                tx[i] = pts[i][j];
                ty[i] = (float)comp;
                xbar += tx[i]; ybar += ty[i];
            }
            xbar /= np_; ybar /= np_;
            double sxx = 0.0, sxy = 0.0;
            for (int i = 0; i < np_; i++) {
                double dx = tx[i] - xbar;
                sxx += dx * dx;
                sxy += dx * (ty[i] - ybar);
            }
            if (sxx <= 0.0) continue;
            double slope = sxy / sxx;
            if (fabs(slope) < 1e-9) continue;
            double intercept = ybar - slope * xbar;
            scale[j]  = slope;
            offset[j] = -intercept / slope;
        }
    }

    /* residual: mean |‖corrected‖ - 1| * 1000 (mg) */
    double res = 0.0;
    for (int i = 0; i < np_; i++) {
        double cx = ((double)pts[i][0] - offset[0]) * scale[0];
        double cy = ((double)pts[i][1] - offset[1]) * scale[1];
        double cz = ((double)pts[i][2] - offset[2]) * scale[2];
        res += fabs(sqrt(cx*cx + cy*cy + cz*cz) - 1.0);
    }
    for (int j = 0; j < 3; j++) { cal.offset[j] = (float)offset[j]; cal.scale[j] = (float)scale[j]; }
    cal.converged   = true;
    cal.residual_mg = (float)(res / (double)np_ * 1000.0);
    return cal;
}

float aa_enmo_sample(float ax, float ay, float az)
{
    float mag = sqrtf(ax * ax + ay * ay + az * az) - 1.0f;
    return mag > 0.0f ? mag : 0.0f;
}

float aa_enmo_sample_cal(float ax, float ay, float az, const aa_calib_t *cal)
{
    if (cal != NULL && cal->converged) {
        ax = (ax - cal->offset[0]) * cal->scale[0];
        ay = (ay - cal->offset[1]) * cal->scale[1];
        az = (az - cal->offset[2]) * cal->scale[2];
    }
    return aa_enmo_sample(ax, ay, az);
}

int aa_epochs_from_accel(const float *accel_xyz, int n_samples, float fs,
                         float *epoch_enmo, float *epoch_t0, int max_epochs)
{
    int w = (int)lroundf(AA_EPOCH_S * fs); if (w < 1) w = 1;
    int n_ep = n_samples / w;
    int out = 0;
    for (int i = 0; i < n_ep && out < max_epochs; i++) {
        double sum = 0.0;
        int base = i * w;
        for (int k = 0; k < w; k++) {
            const float *s = &accel_xyz[(base + k) * 3];
            sum += (double)aa_enmo_sample(s[0], s[1], s[2]);
        }
        epoch_enmo[out] = (float)(sum / (double)w);
        epoch_t0[out]   = (float)i * AA_EPOCH_S;
        out++;
    }
    return out;
}

/* bouts_from_epochs — runs of above-threshold epochs, tolerating brief dips.
 * Faithful to Python: accumulate epochs, allow below-threshold run up to
 * gap_tol, trim trailing ~zero epochs, keep only bouts >= min_s. */
int aa_bouts_from_epochs(const float *epoch_enmo, const float *epoch_t0,
                         int n_epochs, float epoch_dt,
                         aa_bout_t *bouts, int max_bouts)
{
    int nb = 0;
    int cur_start = -1, cur_last = -1;   /* cur_last = last APPENDED epoch idx */
    double cur_sum = 0.0; int cur_cnt = 0;
    float below_run = 0.0f;

    /* close: trim trailing near-zero epochs (Python _close pops enmo<1e-12),
     * then bout end = end of the last remaining appended epoch. */
    #define AA_CLOSE_BOUT() do {                                             \
        int last = cur_last;                                                 \
        double sum = cur_sum; int cnt = cur_cnt;                             \
        while (last >= cur_start && epoch_enmo[last] < 1e-12f) {             \
            sum -= epoch_enmo[last]; cnt--; last--;                          \
        }                                                                    \
        if (cnt > 0 && cur_start >= 0 && last >= cur_start) {                \
            float t0 = epoch_t0[cur_start];                                  \
            float t1 = epoch_t0[last] + epoch_dt;                            \
            if ((t1 - t0) >= AA_BOUT_MIN_S && nb < max_bouts) {              \
                bouts[nb].t_start = t0;                                      \
                bouts[nb].t_end   = t1;                                      \
                bouts[nb].mean_enmo_g = (float)(sum / (double)cnt);          \
                bouts[nb].n_epochs = cnt;                                    \
                nb++;                                                        \
            }                                                                \
        }                                                                    \
        cur_start = -1; cur_last = -1; cur_sum = 0.0; cur_cnt = 0;           \
        below_run = 0.0f;                                                    \
    } while (0)

    for (int i = 0; i < n_epochs; i++) {
        float e = epoch_enmo[i];
        if (e >= AA_BOUT_THRESHOLD_G) {
            if (cur_start < 0) cur_start = i;
            cur_last = i;
            cur_sum += e; cur_cnt++;
            below_run = 0.0f;
        } else if (cur_start >= 0) {
            below_run += epoch_dt;
            if (below_run <= AA_BOUT_GAP_TOL_S) {
                cur_last = i;                 /* dip epoch is appended too */
                cur_sum += e; cur_cnt++;
            } else {
                AA_CLOSE_BOUT();
            }
        }
    }
    if (cur_start >= 0) AA_CLOSE_BOUT();
    #undef AA_CLOSE_BOUT
    return nb;
}

/* ======================= Feature 1: RR irregularity ====================== */
aa_feat_status_t aa_rr_irregularity(const float *rr_dur_s, const bool *rr_gap,
                                    int n, float *burden, int *n_valid)
{
    static float d[AA_MAX_RR];
    int m = 0;
    for (int i = 0; i < n && m < AA_MAX_RR; i++) {
        if (!rr_gap[i] && rr_dur_s[i] >= AA_RR_MIN_S && rr_dur_s[i] <= AA_RR_MAX_S)
            d[m++] = rr_dur_s[i];
    }
    *n_valid = m;
    if (m < AA_RR_MIN_INTERVALS) return AA_FEAT_REFUSED;

    int half = AA_RR_WINDOW_BEATS / 2;
    int n_irreg = 0;
    static float wbuf[AA_RR_WINDOW_BEATS + 1];
    for (int i = 0; i < m; i++) {
        int lo = i - half; if (lo < 0) lo = 0;
        int hi = i + half + 1; if (hi > m) hi = m;
        int wc = 0;
        for (int j = lo; j < hi; j++) wbuf[wc++] = d[j];
        float med = aa_median(wbuf, wc);      /* aa_median sorts wbuf (a copy) */
        if (med > 0.0f && fabsf(d[i] - med) > AA_RR_DEV_FRAC * med) n_irreg++;
    }
    *burden = (float)n_irreg / (float)m;
    return AA_FEAT_OK;
}

/* ======================= Feature 2: ATC95 ================================ */
aa_feat_status_t aa_atc95(const float *bout_mean_enmo, int n_bouts, float *value)
{
    if (n_bouts < AA_ATC_MIN_BOUTS) return AA_FEAT_REFUSED;
    static float m[AA_MAX_BOUTS];
    int nn = n_bouts > AA_MAX_BOUTS ? AA_MAX_BOUTS : n_bouts;
    for (int i = 0; i < nn; i++) m[i] = bout_mean_enmo[i];
    *value = aa_percentile(m, nn, AA_ATC_PERCENTILE);
    return AA_FEAT_OK;
}

/* ======================= Feature 6: HRR60 ================================ */
bool aa_hr_at(const float *beat_times, int n_beats, float t_centre,
              float window_s, float max_gap_s, float *hr)
{
    float lo = t_centre - window_s / 2.0f;
    float hi = t_centre + window_s / 2.0f;
    /* collect beats in window (assumed sorted) */
    float prev = 0.0f; bool have_prev = false;
    double sum_int = 0.0; int n_int = 0; float max_int = 0.0f; int n_in = 0;
    for (int i = 0; i < n_beats; i++) {
        float t = beat_times[i];
        if (t < lo) continue;
        if (t > hi) break;
        n_in++;
        if (have_prev) {
            float d = t - prev;
            sum_int += d; n_int++;
            if (d > max_int) max_int = d;
        }
        prev = t; have_prev = true;
    }
    if (n_in < 2) return false;
    if (max_int > max_gap_s) return false;
    if (n_int == 0) return false;
    float mean_int = (float)(sum_int / (double)n_int);
    if (mean_int <= 0.0f) return false;
    *hr = 60.0f / mean_int;
    return true;
}

static bool aa__bout_qualifies(const float *bt, int nb, const aa_bout_t *bout,
                               float *hr_peak)
{
    if ((bout->t_end - bout->t_start) < AA_HRR_MIN_BOUT_S) return false;
    float hr_end;
    if (!aa_hr_at(bt, nb, bout->t_end, AA_HRR_HR_WINDOW_S, AA_HRR_MAX_GAP_S, &hr_end))
        return false;
    float t_prev = bout->t_end - AA_HRR_HR_WINDOW_S;
    if (t_prev <= bout->t_start) return false;
    float hr_prev;
    if (!aa_hr_at(bt, nb, t_prev, AA_HRR_HR_WINDOW_S, AA_HRR_MAX_GAP_S, &hr_prev))
        return false;
    if (fabsf(hr_end - hr_prev) > AA_HRR_SETTLE_TOL_BPM) return false;
    if (hr_end < AA_HRR_MIN_PEAK_BPM) return false;
    *hr_peak = hr_end;
    return true;
}

aa_feat_status_t aa_hrr60(const float *beat_times, int n_beats,
                          const aa_bout_t *bouts, int n_bouts, float *hrr_bpm)
{
    if (n_beats < 2 || n_bouts < 1) return AA_FEAT_REFUSED;
    float last_bt = beat_times[n_beats - 1];
    float best = -1e30f; bool found = false;
    for (int b = 0; b < n_bouts; b++) {
        float hr_peak;
        if (!aa__bout_qualifies(beat_times, n_beats, &bouts[b], &hr_peak)) continue;
        float t_rec = bouts[b].t_end + AA_HRR_RECOVERY_S;
        if (t_rec + AA_HRR_HR_WINDOW_S / 2.0f > last_bt) continue;
        float hr_rec;
        if (!aa_hr_at(beat_times, n_beats, t_rec, AA_HRR_HR_WINDOW_S,
                      AA_HRR_MAX_GAP_S, &hr_rec)) continue;
        float drop = hr_peak - hr_rec;
        if (drop > best) { best = drop; found = true; }
    }
    if (!found) return AA_FEAT_REFUSED;
    *hrr_bpm = best;
    return AA_FEAT_OK;
}

/* ======================= Feature 5: HRACS ================================ */
aa_feat_status_t aa_hracs_slope(const float *intensity, const float *hr,
                                int n, float *slope, float *r2)
{
    /* keep finite, hr>0 (mirror np filter) */
    static float xs[AA_MAX_HR_PTS], ys[AA_MAX_HR_PTS];
    int m = 0;
    for (int i = 0; i < n && m < AA_MAX_HR_PTS; i++) {
        if (isfinite(intensity[i]) && isfinite(hr[i]) && hr[i] > 0.0f) {
            xs[m] = intensity[i]; ys[m] = hr[i]; m++;
        }
    }
    if (m < AA_HRACS_MIN_POINTS) return AA_FEAT_REFUSED;
    float xmin = xs[0], xmax = xs[0];
    for (int i = 1; i < m; i++) { if (xs[i] < xmin) xmin = xs[i]; if (xs[i] > xmax) xmax = xs[i]; }
    if ((xmax - xmin) < AA_HRACS_MIN_INTENSITY_RANGE_G) return AA_FEAT_REFUSED;

    /* least-squares slope/intercept (double accumulators) */
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < m; i++) { sx += xs[i]; sy += ys[i]; sxx += (double)xs[i]*xs[i]; sxy += (double)xs[i]*ys[i]; }
    double denom = (double)m * sxx - sx * sx;
    if (fabs(denom) < 1e-12) return AA_FEAT_REFUSED;
    double slp = ((double)m * sxy - sx * sy) / denom;
    double icpt = (sy - slp * sx) / (double)m;

    double ss_res = 0, ss_tot = 0, ybar = sy / (double)m;
    for (int i = 0; i < m; i++) {
        double pred = slp * xs[i] + icpt;
        ss_res += (ys[i] - pred) * (ys[i] - pred);
        ss_tot += (ys[i] - ybar) * (ys[i] - ybar);
    }
    *slope = (float)slp;
    *r2 = (ss_tot > 0.0) ? (float)(1.0 - ss_res / ss_tot) : 0.0f;
    return AA_FEAT_OK;
}

aa_feat_status_t aa_hracs(const float *beat_times, int n_beats,
                          const float *enmo_series, int enmo_n, float enmo_fs,
                          float t_start, float t_end, float *slope, float *r2)
{
    if (n_beats < 1) return AA_FEAT_REFUSED;
    if (t_end <= 0.0f) {
        float be = beat_times[n_beats - 1];
        float ee = (float)enmo_n / enmo_fs;
        t_end = be < ee ? be : ee;
    }
    static float xs[AA_MAX_HR_PTS], ys[AA_MAX_HR_PTS];
    int m = 0;
    float t = t_start;
    while (t + AA_HRACS_WINDOW_S <= t_end && m < AA_MAX_HR_PTS) {
        float tc = t + AA_HRACS_WINDOW_S / 2.0f;
        int i0 = (int)(t * enmo_fs);
        int i1 = (int)((t + AA_HRACS_WINDOW_S) * enmo_fs);
        if (i1 > enmo_n) break;
        double s = 0.0;
        for (int k = i0; k < i1; k++) s += enmo_series[k];
        float x = (float)(s / (double)(i1 - i0));
        float hr;
        if (aa_hr_at(beat_times, n_beats, tc, AA_HRACS_WINDOW_S,
                     AA_HRR_MAX_GAP_S, &hr)) {
            xs[m] = x; ys[m] = hr; m++;
        }
        t += AA_HRACS_WINDOW_S;
    }
    return aa_hracs_slope(xs, ys, m, slope, r2);
}

/* ======================= Features 3 & 4 ================================== */
aa_feat_status_t aa_dipping(float dtau_day_ms, int n_day,
                            float dtau_night_ms, int n_night, float *frac)
{
    if (n_day < AA_DIP_MIN_BEATS_PER_PHASE) return AA_FEAT_REFUSED;
    if (n_night < AA_DIP_MIN_BEATS_PER_PHASE) return AA_FEAT_REFUSED;
    if (dtau_day_ms == 0.0f) return AA_FEAT_REFUSED;
    *frac = (dtau_day_ms - dtau_night_ms) / fabsf(dtau_day_ms);
    return AA_FEAT_OK;
}

aa_feat_status_t aa_reserve(float dtau_rest_ms, int n_rest,
                            float dtau_active_ms, int n_active, float *delta_ms)
{
    if (n_rest < AA_RESERVE_MIN_BEATS_PER_PHASE) return AA_FEAT_REFUSED;
    if (n_active < AA_RESERVE_MIN_BEATS_PER_PHASE) return AA_FEAT_REFUSED;
    *delta_ms = dtau_rest_ms - dtau_active_ms;
    return AA_FEAT_OK;
}