/* ============================================================================
 * arterialage.c — faithful C port of the validated arterialage pipeline.
 * Math mirrors the Python reference exactly; only language + fixed buffers
 * differ. See arterialage.h for the pipeline overview.
 * ========================================================================== */
#include "arterialage.h"
#include <math.h>
#include <string.h>

/* ===== frozen band-pass SOS coefficients ================================= *
 * Designed offline: scipy.signal.butter(4,[0.5,15],btype='band',fs=1024,
 * output='sos'), then a0-normalized so the runtime loop has no divides.
 * Order-4 band-pass => 4 sections. These are CONSTANTS baked into firmware,
 * exactly the dsp.py "coefficients FROZEN into firmware" note.
 * Each row: b0, b1, b2, a1, a2  (a0 already divided out).
 * ========================================================================== */
static const float AA_SOS[AA_SOS_SECTIONS][5] = {
    { 3.4955344518e-06f,  6.9910689037e-06f,  3.4955344518e-06f, -1.8463580912e+00f,  8.5333156223e-01f },
    { 1.0000000000e+00f,  2.0000000000e+00f,  1.0000000000e+00f, -1.9282669588e+00f,  9.3632046161e-01f },
    { 1.0000000000e+00f, -2.0000000000e+00f,  1.0000000000e+00f, -1.9940639266e+00f,  9.9407448561e-01f },
    { 1.0000000000e+00f, -2.0000000000e+00f,  1.0000000000e+00f, -1.9977586214e+00f,  9.9776818953e-01f },
};

/* ========================================================================= *
 *  StreamingBiquad  (streaming.py)
 * ========================================================================= */
void aa_biquad_init(aa_biquad_t *bq)
{
    for (int s = 0; s < AA_SOS_SECTIONS; s++) {
        bq->b0[s] = AA_SOS[s][0];
        bq->b1[s] = AA_SOS[s][1];
        bq->b2[s] = AA_SOS[s][2];
        bq->a1[s] = AA_SOS[s][3];
        bq->a2[s] = AA_SOS[s][4];
    }
    aa_biquad_reset(bq);
}

void aa_biquad_reset(aa_biquad_t *bq)
{
    for (int s = 0; s < AA_SOS_SECTIONS; s++) { bq->z1[s] = 0.0f; bq->z2[s] = 0.0f; }
}

/* transposed Direct-Form II, one sample; identical recurrence to Python push().
 * Arithmetic in double to match the float64 reference and stop IIR drift. */
float aa_biquad_push(aa_biquad_t *bq, float x)
{
    double y = (double)x;
    for (int s = 0; s < AA_SOS_SECTIONS; s++) {
        double out = (double)bq->b0[s] * y + bq->z1[s];
        bq->z1[s] = (double)bq->b1[s] * y - (double)bq->a1[s] * out + bq->z2[s];
        bq->z2[s] = (double)bq->b2[s] * y - (double)bq->a2[s] * out;
        y = out;
    }
    return (float)y;
}

/* ========================================================================= *
 *  StreamingPeakDetector  (streaming.py) — THE C REFERENCE, faithful port
 * ========================================================================= */
static int32_t aa__imax(const float *a, int n)   /* argmax, first-max like np */
{
    int32_t k = 0; float m = a[0];
    for (int i = 1; i < n; i++) if (a[i] > m) { m = a[i]; k = i; }
    return k;
}

void aa_peakdet_init(aa_peakdet_t *d, float fs)
{
    d->fs = fs;
    d->thr_frac = AA_THR_FRAC_RHYTHM;
    d->refr   = (int32_t)(AA_REFRACTORY_S * fs); if (d->refr < 1) d->refr = 1;
    d->search = (int32_t)(AA_SEARCH_S * fs);     if (d->search < 1) d->search = 1;
    if (d->search > AA_SEARCH_MAX) d->search = AA_SEARCH_MAX;
    d->env_decay = expf(-1.0f / (AA_ENV_TAU_S * fs > 1e-9f ? AA_ENV_TAU_S * fs : 1e-9f));
    d->env_rise  = AA_ENV_RISE;
    d->seed_n    = (int32_t)(AA_SEED_S * fs); if (d->seed_n < 1) d->seed_n = 1;
    aa_peakdet_reset(d);
}

void aa_peakdet_reset(aa_peakdet_t *d)
{
    d->env = 0.0f;
    d->last_idx = -d->refr - 1;
    d->n = -1;
    d->mode = AA_IDLE;
    d->search_i = 0;
    d->cross_idx = -1;
    d->skip_until = -1;
    d->prev_x = 0.0f;
    d->seq = 0;
    d->seed_max = 0.0f;
    d->seeded = false;
    d->last_beat_idx = -1;
    d->last_beat_amp = 0.0f;
}

/* internal: emit — mirrors Python _emit(); sets last_beat_* and returns idx */
static int32_t aa__emit(aa_peakdet_t *d)
{
    int32_t k_off = aa__imax(d->search_buf, d->search_i);
    int32_t k = d->cross_idx + k_off;
    float amp = d->search_buf[k_off];
    d->env = (1.0f - d->env_rise) * d->env + d->env_rise * amp;
    d->last_idx = k;
    d->skip_until = k + d->refr;
    d->mode = AA_SKIP;
    d->last_beat_idx = k;
    d->last_beat_amp = amp;
    d->seq += 1;
    d->search_i = 0;
    return k;
}

bool aa_peakdet_push(aa_peakdet_t *d, float x, int32_t *beat_idx)
{
    d->n += 1;
    int32_t n = d->n;

    /* envelope seeding: first seed_n samples establish the scale */
    if (!d->seeded) {
        float ax = fabsf(x);
        if (ax > d->seed_max) d->seed_max = ax;
        if (n >= d->seed_n - 1) {
            d->env = d->seed_max > 0.0f ? d->seed_max : 1.0f;
            d->seeded = true;
        }
        d->prev_x = x;
        return false;
    }

    if (d->mode == AA_SKIP) {
        if (n >= d->skip_until) d->mode = AA_IDLE;
        d->prev_x = x;
        return false;
    }

    if (d->mode == AA_SEARCH) {
        d->search_buf[d->search_i] = x;
        d->search_i += 1;
        if (d->search_i >= d->search) { *beat_idx = aa__emit(d); return true; }
        d->prev_x = x;
        return false;
    }

    /* IDLE */
    {
        float thr = d->thr_frac * d->env;
        bool rising = (x > thr) && (d->prev_x <= thr);
        if (rising && (n - d->last_idx) > d->refr) {
            d->mode = AA_SEARCH;
            d->cross_idx = n;
            d->search_i = 1;
            d->search_buf[0] = x;
            d->prev_x = x;
            if (d->search == 1) { *beat_idx = aa__emit(d); return true; }
            return false;
        }
        float decayed = d->env * d->env_decay;
        d->env = x > decayed ? x : decayed;
        d->prev_x = x;
        return false;
    }
}

bool aa_peakdet_flush(aa_peakdet_t *d, int32_t *beat_idx)
{
    if (d->mode == AA_SEARCH && d->search_i > 0) { *beat_idx = aa__emit(d); return true; }
    return false;
}

/* ========================================================================= *
 *  aa_detect_peaks — bandpass then run detector over a buffer
 *  (adaptive_threshold_peaks + dtau_pipeline bandpass, combined)
 * ========================================================================= */
int aa_detect_peaks(const float *raw, int n, float fs,
                    float *ac_out, int32_t *peak_idx, int max_peaks)
{
    aa_biquad_t bq; aa_biquad_init(&bq);
    for (int i = 0; i < n; i++) ac_out[i] = aa_biquad_push(&bq, raw[i]);

    aa_peakdet_t d; aa_peakdet_init(&d, fs);
    int count = 0;
    int32_t bi;
    for (int i = 0; i < n; i++) {
        if (aa_peakdet_push(&d, ac_out[i], &bi)) {
            if (count < max_peaks) peak_idx[count++] = bi;
        }
    }
    if (aa_peakdet_flush(&d, &bi)) {
        if (count < max_peaks) peak_idx[count++] = bi;
    }
    return count;
}

/* ========================================================================= *
 *  quality gate (quality.py) — PI = AC/DC on RAW counts
 * ========================================================================= */
float aa_perfusion_index(const float *raw, int n, float fs, int peak_idx)
{
    int lookback = (int)lroundf(1.0f * fs);
    int lo = peak_idx - lookback; if (lo < 0) lo = 0;
    int hi = peak_idx + 1;        if (hi > n) hi = n;
    int seg = hi - lo;
    if (seg < 3) return 0.0f;
    float sum = 0.0f, mn = raw[lo], mx = raw[lo];
    for (int i = lo; i < hi; i++) {
        float v = raw[i];
        sum += v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    float dc = sum / (float)seg;
    if (dc <= 0.0f) return 0.0f;
    return (mx - mn) / dc;
}

bool aa_gate_beat(const float *timing_raw, int n, float fs, int peak_idx,
                  const float *red_raw, const float *ir_raw)
{
    float pi = aa_perfusion_index(timing_raw, n, fs, peak_idx);
    if (pi < AA_PI_MIN_FRAC) return false;
    if (red_raw != NULL && ir_raw != NULL) {
        float pi_red = aa_perfusion_index(red_raw, n, fs, peak_idx);
        float pi_ir  = aa_perfusion_index(ir_raw,  n, fs, peak_idx);
        if (pi_ir > 0.0f) {
            float ratio = pi_red / pi_ir;
            if (!(ratio >= AA_REDIR_RATIO_LO && ratio <= AA_REDIR_RATIO_HI))
                return false;
        }
    }
    return true;
}

/* ========================================================================= *
 *  dtau_ref L1 — foot via intersecting tangents (features/dtau_ref.py)
 * ========================================================================= */
/* least-squares line fit y = m*x + c over points x[a..b], returns m,c.
 * Sums in double: at sample index ~thousands, x*x needs >7 sig digits, so
 * float32 accumulation diverges from numpy's float64 polyfit. */
static void aa__polyfit1(const float *sig, int a, int b, float *m, float *c)
{
    int cnt = b - a + 1;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = a; i <= b; i++) {
        double x = (double)i, y = (double)sig[i];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    double denom = (double)cnt * sxx - sx * sx;
    if (fabs(denom) < 1e-12) { *m = 0.0f; *c = (cnt > 0) ? (float)(sy / cnt) : 0.0f; return; }
    double mm = ((double)cnt * sxy - sx * sy) / denom;
    *m = (float)mm;
    *c = (float)((sy - mm * sx) / cnt);
}

int aa_foot_tangent(const float *sig, int n, float fs,
                    const int32_t *peak_idx, int n_peaks, float *feet_s)
{
    int base_n = (int)(AA_DTAU_BASELINE_MS * 1e-3f * fs); if (base_n < 2) base_n = 2;
    int back   = (int)(0.4f * fs);
    int nf = 0;
    for (int p = 0; p < n_peaks; p++) {
        int pk = peak_idx[p];
        int lo = pk - back; if (lo < 0) lo = 0;
        if (pk - lo < 3) continue;
        /* trough = argmin over sig[lo..pk] */
        int tr = lo; float mn = sig[lo];
        for (int i = lo; i <= pk; i++) if (sig[i] < mn) { mn = sig[i]; tr = i; }
        if (pk - tr < 3) continue;
        int rise = pk - tr;
        int a = tr + (int)(AA_DTAU_FOOT_UPSTROKE_FRAC * rise);
        int b = tr + (int)((1.0f - AA_DTAU_FOOT_UPSTROKE_FRAC) * rise);
        if (b - a < 2) { a = tr; b = pk; }
        if (b - a < 1) continue;
        float m1, c1; aa__polyfit1(sig, a, b, &m1, &c1);   /* upstroke line */
        float m0, c0;
        int b0 = tr - base_n; if (b0 < 0) b0 = 0;
        if (tr - b0 >= 1) aa__polyfit1(sig, b0, tr, &m0, &c0);   /* baseline */
        else { m0 = 0.0f; c0 = sig[tr]; }
        float foot_x;
        if (fabsf(m1 - m0) < 1e-12f) foot_x = (float)tr;
        else foot_x = (float)(((double)c0 - (double)c1) / ((double)m1 - (double)m0));
        if (foot_x < (float)lo || foot_x > (float)pk) foot_x = (float)tr;
        feet_s[nf++] = foot_x / fs;
    }
    return nf;
}

/* ========================================================================= *
 *  helpers: in-place ascending sort + median (small arrays, insertion sort)
 * ========================================================================= */
static void aa__sort(float *a, int n)
{
    for (int i = 1; i < n; i++) {
        float k = a[i]; int j = i - 1;
        while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
        a[j + 1] = k;
    }
}
static float aa__median_sorted(const float *a, int n)
{
    if (n == 0) return 0.0f;
    if (n & 1) return a[n / 2];
    return 0.5f * (a[n / 2 - 1] + a[n / 2]);
}

/* ========================================================================= *
 *  dtau_ref L2 (pair_feet) + L3 (robust_dtau) — features/dtau_ref.py
 * ========================================================================= */
aa_dtau_status_t aa_dtau_from_feet(const float *feet_l_in, int nL,
                                   const float *feet_r_in, int nR,
                                   float *dtau_ms, int *n_final)
{
    /* copy + sort both foot arrays (Python sorts before pairing) */
    static float fl[AA_MAX_BEATS], fr[AA_MAX_BEATS];
    if (nL > AA_MAX_BEATS) nL = AA_MAX_BEATS;
    if (nR > AA_MAX_BEATS) nR = AA_MAX_BEATS;
    for (int i = 0; i < nL; i++) fl[i] = feet_l_in[i];
    for (int i = 0; i < nR; i++) fr[i] = feet_r_in[i];
    aa__sort(fl, nL);
    aa__sort(fr, nR);

    /* L2: nearest-within-window pairing, dtau in ms = (right-left)*1000 */
    static float dtaus[AA_MAX_BEATS];
    static bool used_r[AA_MAX_BEATS];
    for (int i = 0; i < nR; i++) used_r[i] = false;
    float win = AA_DTAU_PAIR_WINDOW_MS * 1e-3f;
    int nd = 0;
    int j = 0;
    if (nR > 0) {
        for (int i = 0; i < nL; i++) {
            float tl = fl[i];
            while (j + 1 < nR && fabsf(fr[j + 1] - tl) < fabsf(fr[j] - tl)) j++;
            if (fabsf(fr[j] - tl) <= win && !used_r[j]) {
                if (nd < AA_MAX_BEATS) dtaus[nd++] = (fr[j] - tl) * 1000.0f;
                used_r[j] = true;
            }
        }
    }

    /* implied HR from left-foot spacing (slip guard) */
    float implied_hr = 60.0f;
    if (nL > 1) {
        static float ibi[AA_MAX_BEATS];
        int ni = 0;
        for (int i = 1; i < nL; i++) ibi[ni++] = fl[i] - fl[i - 1];
        aa__sort(ibi, ni);
        float med_ibi = aa__median_sorted(ibi, ni);
        implied_hr = (med_ibi > 0.0f) ? 60.0f / med_ibi : 0.0f;
    }

    /* L3: hard-reject -> MAD -> median -> refuse */
    if (nd == 0) return AA_DTAU_NO_PAIRS;

    static float d1[AA_MAX_BEATS];
    int n1 = 0;
    for (int i = 0; i < nd; i++)
        if (fabsf(dtaus[i]) <= AA_DTAU_MAX_ABS_MS) d1[n1++] = dtaus[i];
    if (n1 == 0) return AA_DTAU_ALL_BEYOND_CUTOFF;

    static float d1s[AA_MAX_BEATS];
    for (int i = 0; i < n1; i++) d1s[i] = d1[i];
    aa__sort(d1s, n1);
    float med = aa__median_sorted(d1s, n1);

    /* MAD = median(|d1 - med|) */
    static float absdev[AA_MAX_BEATS];
    for (int i = 0; i < n1; i++) absdev[i] = fabsf(d1[i] - med);
    aa__sort(absdev, n1);
    float mad = aa__median_sorted(absdev, n1);

    static float d2[AA_MAX_BEATS];
    int n2 = 0;
    if (mad > 0.0f) {
        for (int i = 0; i < n1; i++)
            if (fabsf(d1[i] - med) <= AA_DTAU_MAD_K * mad) d2[n2++] = d1[i];
    } else {
        for (int i = 0; i < n1; i++) d2[n2++] = d1[i];
    }

    if (n2 < AA_DTAU_MIN_BEATS) { *n_final = n2; return AA_DTAU_TOO_FEW_BEATS; }

    aa__sort(d2, n2);
    float value = aa__median_sorted(d2, n2);

    if (!(implied_hr >= AA_DTAU_MIN_HR_BPM && implied_hr <= AA_DTAU_MAX_HR_BPM)) {
        *n_final = n2;
        return AA_DTAU_HR_IMPLAUSIBLE;
    }

    *dtau_ms = value;
    *n_final = n2;
    return AA_DTAU_OK;
}

/* ========================================================================= *
 *  end-to-end: raw green (both ears) -> dtau ms
 * ========================================================================= */
aa_dtau_status_t aa_dtau_from_green(const float *green_l_raw,
                                    const float *green_r_raw, int n, float fs,
                                    float *ac_l, float *ac_r,
                                    float *dtau_ms, int *n_final)
{
    static int32_t pk_l[AA_MAX_BEATS], pk_r[AA_MAX_BEATS];
    int nl = aa_detect_peaks(green_l_raw, n, fs, ac_l, pk_l, AA_MAX_BEATS);
    int nr = aa_detect_peaks(green_r_raw, n, fs, ac_r, pk_r, AA_MAX_BEATS);

    /* quality gate on RAW counts; keep survivors' peak indices */
    static int32_t gpk_l[AA_MAX_BEATS], gpk_r[AA_MAX_BEATS];
    int gl = 0, gr = 0;
    for (int i = 0; i < nl; i++)
        if (aa_gate_beat(green_l_raw, n, fs, pk_l[i], NULL, NULL)) gpk_l[gl++] = pk_l[i];
    for (int i = 0; i < nr; i++)
        if (aa_gate_beat(green_r_raw, n, fs, pk_r[i], NULL, NULL)) gpk_r[gr++] = pk_r[i];

    /* feet on the bandpassed signal (same one the peaks came from) */
    static float feet_l[AA_MAX_BEATS], feet_r[AA_MAX_BEATS];
    int nfl = aa_foot_tangent(ac_l, n, fs, gpk_l, gl, feet_l);
    int nfr = aa_foot_tangent(ac_r, n, fs, gpk_r, gr, feet_r);

    return aa_dtau_from_feet(feet_l, nfl, feet_r, nfr, dtau_ms, n_final);
}
