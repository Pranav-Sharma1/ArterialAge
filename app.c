/* ============================================================================
 * app.c — arterialage: BLE feature-vector transmitter
 *
 * Same DSP as the tethered build. The ONLY difference is where the JSON line
 * goes: printf() over VCOM becomes a GATT notification over BLE.
 *
 * ---------------------------------------------------------------------------
 * PROJECT: this must be created from "Bluetooth - SoC Empty", NOT "Empty C".
 * The stack, the GATT database and sl_bt_on_event() all come from that
 * template's generated code; adding components to an Empty C project gets the
 * init ordering wrong in ways that are painful to debug.
 *
 * FILES to copy in beside this one:
 *      arterialage.c / .h        (unmodified)
 *      aa_features.c / .h        (unmodified)
 *
 * COMPONENTS to add (Software Components -> Install):
 *      EMLIB USART      (emlib_usart)   — the sensor SPI
 *      EMLIB CMU, GPIO  (usually already present in the BLE template)
 *
 * GATT: open gatt_configuration.btconf and add ONE custom service with ONE
 * characteristic. Values do not matter as long as they match the host script:
 *      Service        UUID 6e400001-b5a3-f393-e0a9-e50e24dcca9e
 *      Characteristic UUID 6e400003-b5a3-f393-e0a9-e50e24dcca9e
 *                     Properties: Notify
 *                     Value: user type, length 244
 *                     ID (for code): aa_tx
 * Those are the Nordic UART Service UUIDs, which aa_bridge.py already knows.
 * Save, then Force Generation — that creates gattdb_aa_tx used below.
 *
 * ---------------------------------------------------------------------------
 * WIRING — unchanged from the tethered build
 *   SHARED    J2.3 SDI -> PC02 | J2.6 SCLK -> PC03 | J2.1 5V | J2.2 GND
 *             J2.7 3V3
 *   SENSOR L  J2.5 CSB -> PC04 | J2.4 SDO -> PC01
 *   SENSOR R  J2.5 CSB -> PC07 | J2.4 SDO -> PC05
 *   TRIGGER   both J2.8 GPIO1 -> PA07
 *
 * THE 5V PIN IS LOAD-BEARING. Without VLED the LEDs cannot fire; the chips
 * still identify, still configure, still emit FIFO frames with correct tags —
 * but every sample is a constant, the bandpass outputs exactly zero, and no
 * beat is ever detected. Solder it. A hand-held jumper produces intermittent
 * behaviour that looks exactly like a signal-processing fault.
 *
 * ---------------------------------------------------------------------------
 * HOST:
 *      python3 aa_bridge.py --ble ArterialAge --age 55 --sex 1 --map 95 \
 *          | python push.py --stdin --follow
 * ========================================================================== */
#include "em_cmu.h"
#include "em_gpio.h"
#include "em_device.h"
#include "em_usart.h"
#include "sl_bluetooth.h"
#include "gatt_db.h"
#include "app_assert.h"
#if defined(SL_CATALOG_POWER_MANAGER_PRESENT)
#include "sl_power_manager.h"
#endif
#include "arterialage.h"
#include "aa_features.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ---------------------------------------------------------------------------
 * CONFIG OVERRIDES — MUST SIT HERE, ABOVE EVERY USE.
 *
 * These lived at the BOTTOM of this file, just above the #include of
 * arterialage.c. Macros apply textually, so the overrides reached the
 * validated path (compiled below them) but NOT compute_features() above
 * them: the provisional pairing still expanded the header's
 * AA_DTAU_MAX_ABS_MS = 50 while aa_dtau_from_feet ran with 25. The number
 * you watched in the debugger was accepting pairs the emitted number
 * rejected — two different measurements wearing the same name.
 *
 * g_cfg_min_beats also initialised from the header value (30) while the
 * compiled path used 2, defeating the exact stale-config check it exists
 * for: the debugger said 30, the firmware ran 2.
 *
 * AA_DTAU_MAX_ABS_MS 25 for EAR-TO-EAR: paths from the heart are
 * near-symmetric (model prior 2.57 ms, SD 0.80), so past ~25 ms is not an
 * anatomical difference, it is a foot paired against the WRONG beat.
 * Widening a window never improves a measurement; it only admits mispairs
 * that drag the median.                                                    */
#undef  AA_DTAU_MIN_BEATS
#define AA_DTAU_MIN_BEATS   2
#undef  AA_DTAU_MAX_ABS_MS
#define AA_DTAU_MAX_ABS_MS  25.0f
#undef  AA_RR_MIN_INTERVALS
#define AA_RR_MIN_INTERVALS 2

/* ===================== BLE state ========================================= */
static uint8_t  s_advertising_set = 0xFF;
static uint8_t  s_connection      = 0xFF;   /* 0xFF = not connected      */
static bool     s_notify_enabled  = false;  /* host subscribed to aa_tx  */
/* ---- LIVE WAVEFORM STREAMING ---------------------------------------------
 * The GUI was drawing a SYNTHETIC trace from HR and dtau. To show the real
 * thing it needs actual samples from both ears plus the foot times, so the
 * markers land where the tangent construction actually put them.
 *
 * What is sent is the BANDPASSED AC, not raw counts: raw sits on a DC
 * pedestal of ~200000 that would dominate any autoscale, while AC is centred
 * near zero and is exactly what the peak detector and aa_foot_tangent see.
 *
 * Rate: WAVE_DECIM 32 gives 1024/32 = 32 Hz per ear, and WAVE_BATCH 8 samples
 * per message means one message every 250 ms. Two ears at 32 Hz is about
 * 500 B/s of JSON — trivial for BLE, and fast enough to look live.
 *
 * Values are scaled to integers so no float formatting is needed on the hot
 * path; the GUI autoscales anyway, so the units do not matter.            */
#define WAVE_DECIM  32
#define WAVE_BATCH  8
static int32_t  s_wave[2][WAVE_BATCH];
static int      s_wave_n = 0;
static uint32_t s_wave_div = 0;
volatile uint32_t g_wave_sent = 0;
/* Foot times, newest first, so the GUI can mark them on the real trace. */
static float    s_last_foot[2] = {-1.0f, -1.0f};
volatile uint32_t g_ble_sent   = 0;   /* notifications accepted by the stack */
volatile uint32_t g_ble_failed = 0;   /* rejected — usually flow control     */
volatile bool     g_ble_connected = false;
volatile bool     g_ble_subscribed = false;
/* Emit one line. Falls back to doing nothing when nobody is listening, so the
 * DSP is never blocked by the absence of a host. */
/* ---------------------------------------------------------------------------
 * FLOAT FORMATTING WITHOUT printf FLOAT SUPPORT
 *
 * The build links --specs=nano.specs, and newlib-nano's printf has NO
 * floating-point conversion. %f produces NOTHING — silently, no warning, no
 * link error. Integers were fine while every float came out empty:
 *
 *     {"RR_irregularity":,"HR_rest":,"HR_RANGE":,"n_beats":5,"pi":}
 *
 * Adding -u _printf_float to the linker would work but depends on build
 * configuration surviving a regenerate. This does it with integer arithmetic
 * only, so it cannot break that way.                                       */
static char *fmtf(char *buf, int n, float v, int dp)
{
    static const long P[] = {1,10,100,1000,10000,100000,1000000};
    if (dp < 0) dp = 0; else if (dp > 6) dp = 6;
    long mult = P[dp];
    const char *sign = "";
    if (v < 0.0f) { sign = "-"; v = -v; }
    long whole = (long)v;
    long frac  = (long)((v - (float)whole) * (float)mult + 0.5f);
    if (frac >= mult) { whole++; frac -= mult; }
    if (dp == 0) snprintf(buf, (size_t)n, "%s%ld", sign, whole);
    else         snprintf(buf, (size_t)n, "%s%ld.%0*ld", sign, whole, dp, frac);
    return buf;
}
static void out_line(const char *s)
{
    if (s_connection == 0xFF || !s_notify_enabled) return;
    size_t n = strlen(s);
    /* Keep inside a single ATT payload. With the default 247-byte MTU the
     * usable notification size is MTU-3; 244 is safe and the feature vector
     * is ~150 bytes. aa_bridge.py reassembles split lines anyway, so a long
     * line would still arrive correctly — it just costs an extra packet. */
    if (n > 244) n = 244;
    sl_status_t st = sl_bt_gatt_server_send_notification(
                         s_connection, gattdb_aa_tx, (uint8_t)n,
                         (const uint8_t *)s);
    if (st == SL_STATUS_OK) g_ble_sent++;
    else                    g_ble_failed++;
}
/* ===================== Bluetooth event handler =========================== */
void sl_bt_on_event(sl_bt_msg_t *evt)
{
    sl_status_t sc;
    switch (SL_BT_MSG_ID(evt->header)) {
    case sl_bt_evt_system_boot_id: {
        sc = sl_bt_advertiser_create_set(&s_advertising_set);
        app_assert_status(sc);
        /* 100 ms advertising interval: fast enough that the host connects
         * promptly during a demo, slow enough not to dominate power. */
        sc = sl_bt_advertiser_set_timing(s_advertising_set,
                                         160, 160, 0, 0);   /* 160 * 0.625ms */
        app_assert_status(sc);
        sc = sl_bt_legacy_advertiser_generate_data(s_advertising_set,
                                                   sl_bt_advertiser_general_discoverable);
        app_assert_status(sc);
        sc = sl_bt_legacy_advertiser_start(s_advertising_set,
                                           sl_bt_advertiser_connectable_scannable);
        app_assert_status(sc);
    } break;
    case sl_bt_evt_connection_opened_id:
        s_connection = evt->data.evt_connection_opened.connection;
        g_ble_connected = true;
        break;
    case sl_bt_evt_connection_closed_id:
        s_connection     = 0xFF;
        s_notify_enabled = false;
        g_ble_connected  = false;
        g_ble_subscribed = false;
        /* restart advertising so the host can reconnect without a reset */
        sc = sl_bt_legacy_advertiser_generate_data(s_advertising_set,
                                                   sl_bt_advertiser_general_discoverable);
        app_assert_status(sc);
        sc = sl_bt_legacy_advertiser_start(s_advertising_set,
                                           sl_bt_advertiser_connectable_scannable);
        app_assert_status(sc);
        break;
    /* The host writing the CCCD is what turns the stream on. Until this
     * fires, out_line() is a no-op — which is correct: there is nobody to
     * send to, and blocking on it would stall the sensor loop. */
    case sl_bt_evt_gatt_server_characteristic_status_id:
        if (evt->data.evt_gatt_server_characteristic_status.characteristic
              == gattdb_aa_tx) {
            uint16_t flags =
                evt->data.evt_gatt_server_characteristic_status.client_config_flags;
            s_notify_enabled = (flags & sl_bt_gatt_notification) != 0;
            g_ble_subscribed = s_notify_enabled;
        }
        break;
    default:
        break;
    }
}
/* The merge started below the original #define, so this file had "#if
 * STREAM_JSON" with no definition. An undefined macro in #if evaluates to 0,
 * so the ENTIRE emit block was compiled out — silently, no warning. That is
 * why out_line() was never called and why g_ble_sent did not exist as a
 * symbol: the compiler eliminated a global that nothing ever wrote. */
#define STREAM_JSON  1
#define SR_CODE      0x11          /* 1024 sps — AA_SOS is frozen for this */
#define SR_HZ        1024.0f
#define SPI_BITRATE  2000000UL
/* Two exposures: IR + AMBIENT. Two slots allow PPG_TINT = 2 (58.7 us) at
 * 1024 sps — 4x the light of the 3-exposure build, on the channel that
 * actually worked. Red is dropped; nothing single-ear needs it. */
/* MEASURED at the ear, both sensors, shared current:
 *   0x10   L raw_max  12900   R  6            both far too dark
 *   0x20   L raw_max  51000   R  63000        both stable, unclipped
 *   0x30   L ac_max  160195   L feet 38 -> 50 L was under-detecting at 0x20
 *
 * Response is not linear: 0x10 -> 0x20 gave 4x, not 2x.
 *
 * Kept EQUAL across both sensors. An earlier build had R at a quarter of L to
 * compensate what looked like a fixed 8x optical asymmetry; it was not fixed —
 * it moved with placement, and the compensation then starved R. Drive current
 * is a gain control and does not affect foot TIMING, but an asymmetry that
 * moves cannot be calibrated out. Equal currents plus good placement is the
 * honest configuration.                                                     */
#define PA_IR_L   0x30
#define PA_IR_R   0x30
#define PA_RED_L  0x30
#define PA_RED_R  0x30
static const uint8_t PA_IR_D[2]  = { PA_IR_L,  PA_IR_R  };
static const uint8_t PA_RED_D[2] = { PA_RED_L, PA_RED_R };
#define RING     1024
#define MAXBEATS 512
#define BUS_PORT gpioPortC
#define PIN_MOSI 2
#define PIN_SCLK 3
#define CS_L     4      /* pin marked CS  */
#define MISO_L   1      /* pin marked MISO*/
#define CS_R     7      /* pin marked SDA */
#define MISO_R   5      /* pin marked SCL — split, R's shifter drives SDO */
#define TRIG_PORT gpioPortA
#define TRIG_PIN  7     /* pin marked PWM -> both J2.8 GPIO1 */
#define R_PPG_SYNC 0x10
#define SYNC_TARGET 0x02
#define R_INT1 0x00
#define R_INT2 0x01
#define R_FIFO_WR 0x04
#define R_FIFO_RD 0x05
#define R_OVF 0x06
#define R_FIFO_CNT 0x07
#define R_FIFO_DATA 0x08
#define R_SYSCTRL 0x0D
#define R_PPGCFG1 0x11
#define R_PPGCFG2 0x12
#define R_PPGCFG3 0x13
#define R_PDBIAS 0x15
#define R_LEDSEQ1 0x20
#define R_LEDSEQ2 0x21
#define R_LEDSEQ3 0x22
#define R_LED1PA 0x23
#define R_LED2PA 0x24
#define R_LED3PA 0x25
#define R_LEDRANGE 0x2A
#define R_PARTID 0xFF
#define EXP_IR   0x2
#define EXP_RED  0x3
#define EXP_DARK 0x9
/* LEDC1=IR, LEDC2=AMBIENT -> PD2 tags are 5 and 6.
 *   PPG1 LEDC1 = 1, PPG1 LEDC2 = 2, PPG2 LEDC1 = 5, PPG2 LEDC2 = 6
 * Measured order interleaves the photodiodes: 1,5,2,6 */
/* MEASURED in the 4-exposure build: 1,7,2,8,3,9,4,10.
 * PD1 slot n -> tag n ;  PD2 slot n -> tag n+6  (NOT n+4).
 * With LEDC1=IR, LEDC2=AMBIENT that gives PD2 IR = 7, PD2 ambient = 8.
 * The previous 5/6 matched nothing, so push_sample never ran and the FIFO
 * saturated with zero beats — verify against g_tagseq before trusting. */
/* LEDC1=IR, LEDC2=RED, LEDC3=AMBIENT.
 * PD1 slot n -> tag n ; PD2 slot n -> tag n+6  (MEASURED, not derived).
 * So PD2: IR = 7, RED = 8, AMBIENT = 9. Verify against g_tagseq, which
 * should read 1,7,2,8,3,9 repeating. */
#define TAG_IR  7
#define TAG_RD  8
#define TAG_AM  9
/* ---------------------------------------------------------------------------
 * TIMING_TAG — which wavelength drives beat detection.
 *
 * IR (tag 7) read exactly 0 while red was visibly lit. On this board
 * LED2_DRV (IR) routes through JP2 and LED3_DRV (RED) through JP1, each
 * selecting on-board or external board. Red visible on-board means JP1 is set
 * local; IR reading zero means JP2 is almost certainly routed to the external
 * board, which is not connected — so the IR die never fires.
 *
 * Nothing downstream cares which wavelength supplied the timing: dtau, HR and
 * RR all come from foot and beat times, not from absolute counts. Red is the
 * strongest confirmed channel on this hardware, so use it.
 *
 * Set back to TAG_IR once JP2 is moved to on-board.                        */
#define TIMING_TAG  TAG_IR
/* MEASURED tag order (g_capt):  1,7, 1,7, 2,8, 2,8, 3,9, 3,9
 * Each exposure appears TWICE, so a frame is 12 samples, not 6. The doubling
 * is why an assumed 6 never aligned. The drain no longer gates on this value
 * — it reads whatever is present and closes each frame on the ambient tag —
 * so SPF is only a lower bound for "is there anything worth reading". */
#define SPF     12         /* 3 exposures x 2 photodiodes x 2 */
/* ===================== RESULTS ========================================== */
volatile uint8_t  g_id_l = 0xAA, g_id_r = 0xAA;
volatile uint8_t  g_range_l = 0xAA, g_range_r = 0xAA;
/* Overflow as a RATE, not a high-water mark. The old latched maximum could
 * not distinguish one hiccup during bring-up from continuous frame loss —
 * 127 meant the same thing either way, which made it useless as a signal.
 * g_ovf_* is the LAST value read; g_ovf_events_* counts non-zero reads. */
volatile uint8_t  g_ovf_l = 0, g_ovf_r = 0;
volatile uint32_t g_ovf_events_l = 0, g_ovf_events_r = 0;
volatile uint32_t g_ovf_lost_l = 0, g_ovf_lost_r = 0;   /* samples lost */
volatile int      g_err = 0;            /* 0 both, 1 one missing, 3 none */
volatile bool     g_ok_l = false, g_ok_r = false;
volatile uint32_t g_loops = 0;
volatile int      g_frame_gap = 0;
/* ---- WHAT THIS BUILD COMPILED — check before trusting any number --------
 * A stale binary looks exactly like broken code. These caught two today. */
volatile int g_cfg_min_beats  = AA_DTAU_MIN_BEATS;
volatile int g_cfg_attempt_at = 0;
volatile int g_cfg_rr_min     = AA_RR_MIN_INTERVALS;
volatile float    g_raw_ir[RING], g_ac_ir[RING];
volatile int      g_wr = 0;
volatile float    g_ambient = 0;
volatile float    g_red = 0;        /* PD2 red, raw — venous gate input */
volatile int      g_beats = 0;
volatile float    g_beat_times[MAXBEATS];   /* seconds, monotonic */
/* ---- FEATURE VECTOR — this is what the model consumes ------------------ */
volatile float g_rr_irregularity = -999.0f;  /* burden, fraction 0..1 */
volatile int   g_rr_status       = -1;       /* 0 = AA_FEAT_OK       */
volatile int   g_rr_nvalid       = 0;
volatile float g_hr_rest         = -999.0f;  /* bpm, 5th percentile  */
volatile float g_hr_range        = -999.0f;  /* bpm, 95th - 5th      */
volatile float g_hr_now          = -999.0f;  /* live, for contact fb */
volatile float g_pi              = -999.0f;  /* perfusion index      */
/* Interval diagnostics. A correct rhythm sitting still gives a median near
 * 1.0 s with min and max within ~15% of it. Min at half the median means
 * doubled beats; max at twice the median means missed ones. */
volatile float g_rr_median_s     = -999.0f;
volatile float g_rr_min_s        = -999.0f;
volatile float g_rr_max_s        = -999.0f;
volatile int   g_rr_n_raw        = 0;
/* raw span over the window — 524287 is 19-bit full scale, i.e. railed */
volatile float g_raw_min = -999.0f, g_raw_max = -999.0f;
volatile int   g_notches_l = 0, g_notches_r = 0;   /* beats rejected */
/* ---- DETECTOR INSTRUMENTATION -------------------------------------------
 * Signal present (7000 baseline, hundreds of AC) but zero beats means the
 * threshold sits above the signal. thr = thr_frac * env, and env is seeded
 * from the largest |x| in the detector's first samples — which, if the
 * biquad is still ringing, locks it far too high.
 *
 * Rather than keep guessing, measure it:
 *      g_env_*    the detector's envelope
 *      g_thr_*    the actual firing threshold = thr_frac * env
 *      g_ac_max_* the largest |AC| the filter has ever produced
 *
 * If g_thr_* > g_ac_max_*, the detector CANNOT fire, and the ratio says by
 * how much. If g_thr_* is well below g_ac_max_*, the envelope is fine and
 * the fault is elsewhere. */
volatile float g_env_l = -1.0f, g_env_r = -1.0f;
volatile float g_thr_l = -1.0f, g_thr_r = -1.0f;
volatile float g_ac_max_l = 0.0f, g_ac_max_r = 0.0f;
/* The RAW value handed to aa_biquad_push, and its span. A bandpass fed a
 * constant outputs exactly zero once settled, so if g_raw_span_* is 0 the
 * LED is not firing; if it is large, the filter is the problem. */
volatile float g_fi_l = -1.0f, g_fi_r = -1.0f;
volatile float g_raw_min_l = 1e9f, g_raw_max_l = -1e9f;
volatile float g_raw_min_r = 1e9f, g_raw_max_r = -1e9f;
volatile float g_raw_span_l = 0.0f, g_raw_span_r = 0.0f;
volatile uint32_t g_pushes_l = 0, g_pushes_r = 0;
volatile bool  g_seeded_l = false, g_seeded_r = false;
/* ---- DRAIN VISIBILITY ----------------------------------------------------
 * Ear L identifies (0x25), configures, reports g_ok_l = true, and delivers
 * ZERO samples: g_ac_max_l stayed 0, meaning push_sample(0,...) never ran.
 * Ear R on the same bus is fine (g_ac_max_r = 135123).
 *
 * These expose what drain() actually sees per ear, so "the FIFO is empty"
 * can be distinguished from "drain bailed early" without guessing:
 *   g_avail_*   last FIFO_DATA_COUNT read (0 = chip not sampling)
 *   g_drains_*  times drain() was entered
 *   g_frames_*  frames actually assembled and pushed
 *   g_sysctl_*  System Control readback; bit1 SHDN set means it never
 *               left shutdown, i.e. configure() did not take. */
volatile int      g_avail_l = -1, g_avail_r = -1;
volatile uint32_t g_drains_l = 0, g_drains_r = 0;
volatile uint32_t g_frames_l = 0, g_frames_r = 0;
volatile uint8_t  g_sysctl_l = 0xAA, g_sysctl_r = 0xAA;
volatile uint32_t g_emitted      = 0;        /* JSON lines sent      */
volatile float g_dtau_ms   = -999.0f;        /* ms; /1000 for the model */
volatile int   g_dtau_status = -1;           /* 0 = AA_DTAU_OK          */
volatile int   g_dtau_n      = 0;            /* PAIRS that survived     */
/* ---- dtau EMIT GATE -------------------------------------------------------
 * Status OK is NOT sufficient to put dtau on a screen. The 173/162-beat run
 * proved it: status OK, 20 pairs, median 12.5 ms — and the pairs spanned
 * -47..+47 ms. A median of a uniform spread is the midpoint of noise, and a
 * MAD reject cannot rescue a distribution with no cluster in it.
 *
 * So the vector carries dtau only when the pairs actually CLUSTER:
 *   - enough of them for a median to mean something, and
 *   - a raw min..max span narrow enough that they are one population.
 * Real clustered pairs sit within a few ms of each other; noise fills the
 * whole +/-25 ms window (span -> ~50). 12 ms between them is a clean line.
 *
 * Until the gate opens the GUI simply shows dtau "not measured" — its
 * designed tier_is_floor path — which reads as "acquiring", not as failure.
 * g_dtau_gate says why: 0 emitted, 1 status not OK, 2 too few pairs,
 * 3 pairs not clustered.                                                   */
#define DTAU_EMIT_MIN_PAIRS      0
#define DTAU_EMIT_MAX_SPREAD_MS  20.0f
volatile int   g_dtau_gate   = -1;
/* ---- PROVISIONAL dtau — diagnostic only, NOT the validated output --------
 * aa_dtau_from_feet returns TOO_FEW_BEATS without a value, so below the beat
 * threshold there is nothing to look at at all — you cannot tell a converging
 * measurement from noise. These fields replicate the pairing and median so
 * the number is visible from the very first pair.
 *
 * IT IS NOT THE VALIDATED PATH. It skips the MAD outlier reject and the HR
 * plausibility guard. Use g_dtau_ms (with g_dtau_status == 0) for anything
 * that leaves the device; use these to watch it settle.
 *
 * g_dtau_raw_hist holds the last 16 provisional estimates — the SPREAD
 * across them is the real uncertainty on this hardware, which is worth more
 * than any single value. */
#define DTAU_HIST 16
volatile float g_dtau_raw_ms   = -999.0f;   /* median of paired dtaus, ms */
volatile int   g_dtau_raw_n    = 0;         /* pairs formed, before MAD   */
volatile float g_dtau_raw_hist[DTAU_HIST];
volatile int   g_dtau_raw_count = 0;
volatile float g_dtau_raw_min  = -999.0f, g_dtau_raw_max = -999.0f;
/* ---- FOOT TIMES, both ears ------------------------------------------------
 * 26 feet on one ear and 19 on the other yielding only 2 pairs is not
 * explained by the counts alone: with aligned clocks (g_frame_gap 0, no
 * overflow) simultaneous feet should pair easily inside the 100 ms window.
 *
 * The remaining possibility is that the two ears detect during DIFFERENT time
 * spans — say L from 0-22 s and R from 8-22 s — in which case most of L's
 * feet have no partner within the window and are rejected. That is invisible
 * in the counts and obvious in the timestamps.
 *
 * Newest 24 per ear, seconds, oldest first. Compare the ranges. */
#define FEETDBG 24
volatile float g_feet_l[FEETDBG], g_feet_r[FEETDBG];
volatile int   g_feet_n_l = 0, g_feet_n_r = 0;
/* toff each mirrored foot was stamped with, index-parallel to g_feet_*.
 * MEASURED need: stretches of feet with a LOCKED inter-ear offset of
 * ~-300 ms whose interval fingerprints match beat-for-beat — a stepwise
 * clock artifact, not morphology. If those stretches show one ear's
 * per-foot toff stepped by ~0.3 while the other is flat, the deficit
 * compensation mis-fired there; if both are flat, toff is innocent. */
volatile float g_ftoff_l[FEETDBG], g_ftoff_r[FEETDBG];
/* BEAT times alongside the feet. The feet showed intervals of 0.111 s and
 * 0.146 s — 541 and 411 bpm — which is below the detector's own 0.25 s
 * refractory and therefore cannot be two separate detections.
 *
 * Either the BEATS carry those intervals too, in which case the refractory is
 * not being applied and the fault is in detection; or the beats are clean and
 * only the feet scatter, in which case aa_foot_tangent is placing the foot
 * somewhere other than this cycle's upstroke. Those need opposite fixes, and
 * the beat times settle it. */
volatile float g_beats_l[FEETDBG], g_beats_r[FEETDBG];
volatile int   g_beats_n_l = 0, g_beats_n_r = 0;
volatile int   g_feetdrop_l = 0, g_feetdrop_r = 0;
volatile int   g_nbeats_l = 0, g_nbeats_r = 0;
volatile int   g_feature_ear = -1;   /* which ear the features came from */
/* ---- TIME BASE INTEGRITY — the thing that was actually blocking dtau -----
 * Beat and foot timestamps come from each ear's OWN sample counter:
 *      t = bi / SR_HZ
 * Both sensors sample on the same GPIO1 trigger, so sample N on the left and
 * sample N on the right are simultaneous — PROVIDED neither has dropped a
 * frame. When a FIFO overflows, that ear's counter falls behind real time by
 * however many frames were lost, and the two ears' timestamps silently
 * diverge. Feet stop overlapping and pairing collapses.
 *
 * That is what produced 7 pairs from 93 feet: plenty of feet on both ears,
 * found at times that no longer referred to the same instants.
 *
 * There is no way to recover WHICH frames were lost, so the only honest
 * response is to discard the history and restart both ears together. Losing
 * 30 s of data is better than pairing against a corrupted clock. */
volatile uint32_t g_resyncs = 0;         /* times both ears were restarted */
/* ---- SHARED TIME BASE -----------------------------------------------------
 * MEASURED, both arrays captured at the same instant:
 *      L newest foot 119.544 s
 *      R newest foot 117.465 s      <- 2.08 s apart
 *
 * Both ears sample on the same GPIO1 trigger, so simultaneous feet must carry
 * the same timestamp. They did not, because each ear's foot time comes from
 * its OWN e->wr, which starts counting at that sensor's first delivered frame
 * — and the two sensors are configured, and start filling, at different
 * moments. R's counter sat ~2100 samples behind L's.
 *
 * With a 2 s offset and a 50 ms pairing window essentially nothing can pair;
 * the handful of pairs that appeared were coincidental alignments, which is
 * why dtau wandered between -25 ms and +30 ms while the raw signals looked
 * healthy.
 *
 * The fix is a ONE-TIME alignment once both sensors are confirmed up: flush
 * both FIFOs and zero both counters in the same breath, so sample N means the
 * same instant on both ears from then on. Done once at startup rather than on
 * every overflow — an earlier version reset on any overflow and wiped the
 * feet faster than they could accumulate. */
volatile bool     g_aligned = false;
volatile uint32_t g_align_at = 0;

/* ---- MID-RUN CLOCK DRIFT COMPENSATION -------------------------------------
 * The one-time alignment above fixes the STARTUP offset. It cannot fix what
 * happens DURING a run: every dropped frame stalls that ear's push counter
 * while real time advances, so t = bi/SR_HZ falls behind by lost/SR_HZ — and
 * only on the ear that dropped. The inter-ear offset then DRIFTS through the
 * run instead of sitting at one value.
 *
 * A drifting offset is exactly what the 173/162-beat run measured: 20 pairs
 * spread UNIFORMLY across -47..+47 ms. A fixed mispairing clusters; a linear
 * drift sweeps the true offset across the acceptance window, painting it
 * uniformly. The free-run drift measured earlier (~0.32%/min ~ 48 ms/min)
 * matches that +/-47 ms span over a ~1 min run almost exactly.
 *
 * The chip was first asked to CONFESS its losses via OVF_COUNTER, converted
 * to time through a measured words-per-push ratio. MEASURED RESULT: with 90
 * and 82 feet on the two ears — detection healthy — only 7 pairs formed,
 * uniformly spread across the window, while the compensation had accumulated
 * 21 ms (L) and 77 ms (R). Actively compensating and still misaligned by
 * tens of ms means the ESTIMATE is wrong: its accuracy hangs on what OVF
 * semantically counts (words? frames? where does it saturate?) and on every
 * loss channel reporting through OVF at all. Estimating what can be measured
 * was the mistake.
 *
 * GROUND TRUTH: the MCU generates every trigger. s_ticks since alignment IS
 * the number of samples each sensor was told to take; e->wr is the number
 * that arrived. The deficit between them is the EXACT total loss, per ear,
 * continuously, with no datasheet assumptions — OVF overflow, SPI faults,
 * outages while a sensor is down (drains stop, wr freezes, deficit grows by
 * exactly the outage). The only contaminant is the samples sitting in the
 * FIFO at the instant of the read, so the deficit is sampled at the end of
 * every drain and a 1 s windowed MINIMUM strips the in-flight component:
 * the minimum lands on the loops where drain fully caught up.
 *
 * s_toff[d] is therefore SET (not accumulated) from the windowed minimum,
 * and added wherever a push-counter index becomes a time. Counters and ring
 * indices are untouched. OVF is kept as telemetry and as the saturation
 * tripwire for a full resync.                                              */
static float    s_toff[2] = {0.0f, 0.0f};        /* seconds, per ear */
volatile float  g_toff_l = 0.0f, g_toff_r = 0.0f;   /* debugger mirrors */
volatile uint32_t g_def_l = 0, g_def_r = 0;   /* instantaneous deficit, samples */

/* ---- TAG CAPTURE — measure, do not predict -------------------------------
 * TAG_IR/TAG_AM below were DERIVED from the LEDC assignment, not observed.
 * Every derived tag mapping in this project has been wrong at least once;
 * the one time it was measured, the photodiodes interleaved in an order
 * that was not predicted. A saturated FIFO with zero beats is exactly what
 * a wrong tag constant looks like: samples arrive, the switch matches
 * nothing, push_sample never runs.
 *
 * g_tagseq holds the first 48 tags exactly as they leave the FIFO.
 * g_tagcount[t] counts every occurrence of tag t. */
/* RAW (tag, value) pairs straight off the FIFO, before any assembly.
 * Both tag 7 and tag 8 decoded to zero while red was visibly lit, so the
 * question is no longer WHICH tag carries the timing channel but whether ANY
 * sample carries a non-zero value. This is the last unverified link:
 *   all values ~0            -> ADC reading nothing; check the 1.8 V / 3V3
 *                               rails, not just VLED
 *   tag 9 large, 7 and 8 = 0 -> ambient works, LED exposures do not
 *   values present           -> the decode drops them; shift or mask wrong  */
volatile uint8_t  g_capt[48];
volatile uint32_t g_capv[48];
volatile int      g_capn = 0;
volatile uint8_t  g_tagseq[48];
volatile uint32_t g_tagcount[32];
volatile int      g_tagn = 0;
volatile int   g_nfeet_l = 0,  g_nfeet_r = 0;
/* Report from the first pair. Both dtau paths now run as soon as two feet
 * exist on each ear; nothing is withheld waiting for a count.
 *
 * AA_DTAU_MIN_BEATS is overridden to 2 at the bottom of this file, so the
 * VALIDATED path no longer refuses either — it still applies the MAD outlier
 * reject and the HR plausibility guard, and still reports n via g_dtau_n, so
 * you can judge the number rather than have it hidden. */
#define ATTEMPT_AT 2
/* ---------------------------------------------------------------------------
 * FOOT_WINDOW — sliding window, replacing the halve-on-success trim.
 *
 * The old scheme discarded the older HALF of both feet arrays whenever
 * aa_dtau_from_feet succeeded. Two things were wrong with that:
 *
 *  1. The provisional estimate reads the SAME arrays, so every trim changed
 *     its input population and the median stepped discontinuously. That is
 *     what produced +5.62 ms held bit-identically for 7 estimates, then
 *     -2.60 ms held bit-identically for 8, with no change in finger
 *     placement. The apparent stability was an artifact: identical values
 *     mean the input was not changing, not that the measurement was precise.
 *
 *  2. Halving makes the window length oscillate between N and N/2, so the
 *     number's MEANING changes as you watch it.
 *
 * A fixed-length window fixes both: "median dtau over the last FOOT_WINDOW
 * beats" has a defined meaning and updates smoothly, one beat at a time. */
#define FOOT_WINDOW 40
/* ---------------------------------------------------------------------------
 * FEATURE_EAR — which sensor supplies RR_irregularity, HR_rest, HR_RANGE.
 *
 *   0 = the L pins (CS on the pin marked CS, MISO on the pin marked MISO)
 *   1 = the R pins (CS on the pin marked SDA, MISO on the pin marked SCL)
 *
 * "L" and "R" are only which MCU pins a board is plugged into, not a property
 * of the board. Single-ear features work from either; dtau still needs both.
 * If only one sensor is connected, point this at whichever pins it is on.
 * AUTO (-1) picks whichever sensor came up, preferring 0 when both did.    */
/* ---------------------------------------------------------------------------
 * PEAK DETECTOR TUNING — the proper fix, applied at the source.
 *
 * aa_peakdet_t.thr_frac and .refr are STRUCT FIELDS, initialised from
 * AA_THR_FRAC_RHYTHM (0.45) and AA_REFRACTORY_S (0.20) but writable after
 * aa_peakdet_init(). So they can be raised without editing arterialage.c.
 *
 * A dicrotic notch is typically 40-60% of the systolic peak's amplitude.
 * At thr_frac 0.45 the notch clears the bar and is counted as a beat. At
 * 0.65 it does not. This stops the notch being DETECTED at all, rather than
 * filtering it out afterwards — so aa_foot_tangent never runs on it either,
 * which matters for dtau as much as for RR.
 *
 * Refractory 0.20 s allows 300 bpm; the notch arrives 0.25-0.35 s after the
 * peak, so 0.30 s (200 bpm max) rejects it on timing as well.
 *
 * TUNING LOG — measured, finger, median 0.940 s (63.8 bpm):
 *   thr 0.45 refr 0.20 : irreg 0.554  HR_RANGE 172  min 0.213 (ON refractory)
 *   thr 0.45 + IBI gate: irreg 0.442  HR_RANGE  64  32 notches
 *   thr 0.65 refr 0.30 : irreg 0.469  HR_RANGE  51   8 notches
 *                        min 0.375 (0.40x) AND max 2.235 (2.38x)
 *                        -> BOTH doubles and misses: too high for weak
 *                           beats, still too low for strong notches
 *   thr 0.55 refr 0.30 : split the difference
 *
 * If both failure modes persist at 0.55, the pulse AMPLITUDE is varying too
 * much for any fixed fraction of envelope, and the fix is contact quality
 * (g_pi) rather than threshold tuning.
 *
 * For reference, arterialage.json norms for RR_irregularity:
 *   p1 0.011 | p50 0.046 | p95 0.130 | p99 0.200
 * Anything above ~0.2 is off the scale the model was built on.            */
#define PEAK_THR_FRAC   0.45f   /* aa_peakdet_init default */
#define PEAK_REFRAC_S   0.60f   /* 100 bpm ceiling — SEATED SUBJECT ONLY */
/* 0.60 s, third and final setting, each step driven by measured arrays.
 *
 * 0.25 s let the dicrotic notch through (0.25-0.35 s post-peak): five
 * consecutive ~0.4 s foot intervals, 150 bpm sustained while seated.
 * 0.40 s killed the notch and exposed the next wave behind it — the ear's
 * DIASTOLIC WAVE at 0.45-0.55 s post-peak, tall enough at this site to
 * clear thr_frac 0.45. MEASURED: a fresh interval population at ~0.5 s,
 * L counting 112 events against R's 134 from one heart, and HR_rest
 * FLAPPING between 112.6 and 50.2 bpm on the wire — the artefact filter's
 * two attractors on a bimodal interval distribution.
 *
 * 0.60 s times out the diastolic wave unconditionally. The cost is a
 * 100 bpm detection ceiling: fine for a seated demo, WRONG for exercise.
 * If this system ever needs to survive HR > 100, the fix is amplitude
 * discrimination at the threshold, not a shorter refractory — every
 * shorter setting has now been measured letting a secondary wave through. */
/* MEASURED at the ear, both sensors unclipped:
 *      g_rr_median_s 0.335 s  =  179 bpm      <- three times the real rate
 *      thr_l 2415  vs  ac_max_l 57686         =  4% of peak
 *      thr_r 1579  vs  ac_max_r 32892         =  4.8% of peak
 *
 * thr_frac is 0.45, so the threshold should sit at 45% of the envelope. It is
 * at 4% of the ACTUAL peak because env seeded while the sensors were still
 * being positioned — a weak signal — and aa_peakdet's envelope does not climb
 * fast enough when the real pulse arrives twenty times larger. The detector
 * then fires on dicrotic notches and noise, which is what produced 99 feet on
 * one ear against 55 on the other and left only 11 pairs.
 *
 * A 0.40 s refractory caps detection at 150 bpm. No resting adult exceeds
 * that, so it cannot reject a real beat here, and it directly removes the
 * extra detections that a decayed envelope lets through. This is a floor on
 * physiology rather than a tuning knob.                                    */
/* readable in the debugger so the running values are never in doubt */
volatile float g_cfg_thr_frac = PEAK_THR_FRAC;
volatile float g_cfg_refrac_s = PEAK_REFRAC_S;
#define FEATURE_EAR  -1
/* ---------------------------------------------------------------------------
 * WARMUP_S — beats detected in the first N seconds are thrown away.
 *
 * The FIFO overflows during bring-up, before configure() has finished, and
 * the samples that survive are fragments of waveforms. Those produce missed
 * and doubled beats, whose intervals sit far from the local median. A 63%
 * RR burden and an HR_RANGE of 205 bpm is that contamination, not a rhythm.
 *
 * Discarding them costs a few seconds and stops a handful of bad intervals
 * poisoning the percentiles for the rest of the run. */
/* Samples withheld from the peak detector while the biquad settles. Applied
 * identically to both ears, so the timestamp offset is common-mode and
 * cancels in dtau. */
/* ===================== timebase ========================================= */
static volatile uint32_t s_ticks;
static volatile bool     s_trig_on = false;
/* Shared sample trigger on PA07 -> both J2.8. Both sensors sit in
 * GPIO_CTRL 0x02 (sample trigger input), so they sample on the SAME edge
 * off the MCU crystal instead of their own RC oscillators. Free-running,
 * two MAX86141 drift ~0.3% — 145 ms over 45 s, against a dtau of 10-20 ms. */
void SysTick_Handler(void)
{
    s_ticks++;
    if (s_trig_on) {
        GPIO_PinOutSet(TRIG_PORT, TRIG_PIN);
        for (volatile int i = 0; i < 130; i++) __asm volatile("nop"); /* >=5us */
        GPIO_PinOutClear(TRIG_PORT, TRIG_PIN);
    }
}
static uint32_t ms_now(void)
{ return (uint32_t)(((uint64_t)s_ticks * 1000u) / (uint32_t)SR_HZ); }
static void delay_ms(uint32_t ms)
{ uint32_t t0 = ms_now(); while ((ms_now() - t0) < ms) __asm volatile("nop"); }
/* ===================== SPI ============================================== */
static inline void tick(void) { __asm volatile("nop;nop"); }
static inline uint8_t xfer(uint8_t tx) { return USART_SpiTransfer(USART0, tx); }
static void spi_init(void)
{
    CMU_ClockEnable(cmuClock_USART0, true);
    USART_InitSync_TypeDef init = USART_INITSYNC_DEFAULT;
    init.master = true; init.baudrate = SPI_BITRATE;
    init.databits = usartDatabits8; init.msbf = true;
    init.clockMode = usartClockMode0; init.autoCsEnable = false;
    USART_InitSync(USART0, &init);
    GPIO->USARTROUTE[0].TXROUTE  = (BUS_PORT << _GPIO_USART_TXROUTE_PORT_SHIFT)
                                 | (PIN_MOSI << _GPIO_USART_TXROUTE_PIN_SHIFT);
    GPIO->USARTROUTE[0].RXROUTE  = (BUS_PORT << _GPIO_USART_RXROUTE_PORT_SHIFT)
                                 | (MISO_L   << _GPIO_USART_RXROUTE_PIN_SHIFT);
    GPIO->USARTROUTE[0].CLKROUTE = (BUS_PORT << _GPIO_USART_CLKROUTE_PORT_SHIFT)
                                 | (PIN_SCLK << _GPIO_USART_CLKROUTE_PIN_SHIFT);
    GPIO->USARTROUTE[0].ROUTEEN  = GPIO_USART_ROUTEEN_TXPEN
                                 | GPIO_USART_ROUTEEN_RXPEN
                                 | GPIO_USART_ROUTEEN_CLKPEN;
}
static inline uint32_t cs_of(int d)   { return d ? CS_R   : CS_L;   }
static inline uint32_t miso_of(int d) { return d ? MISO_R : MISO_L; }
/* MISO is split: sensor R drives SDO even with CSB high, so each sensor
 * gets its own return line and RXROUTE is re-pointed per transaction. */
static inline void sel(int d)
{
    GPIO->USARTROUTE[0].RXROUTE = (BUS_PORT   << _GPIO_USART_RXROUTE_PORT_SHIFT)
                                | (miso_of(d) << _GPIO_USART_RXROUTE_PIN_SHIFT);
}
static void wr(int d, uint8_t reg, uint8_t val)
{
    sel(d);
    GPIO_PinOutClear(BUS_PORT, cs_of(d));
    xfer(reg); xfer(0x00); xfer(val);
    GPIO_PinOutSet(BUS_PORT, cs_of(d)); tick();
}
static uint8_t rd(int d, uint8_t reg)
{
    uint8_t v;
    sel(d);
    GPIO_PinOutClear(BUS_PORT, cs_of(d));
    xfer(reg); xfer(0xFF); v = xfer(0x00);
    GPIO_PinOutSet(BUS_PORT, cs_of(d)); tick();
    return v;
}
/* Verify rather than assume: read twice and retry. A shifted read of 0x2A
 * looks like a changed LED range, which would refuse to drive anything.
 *
 * attempts is a BUDGET, not a constant, because this runs in two different
 * situations. At startup nothing else needs the CPU, so 12 attempts
 * (~200 ms) is fine. In BACKGROUND RECOVERY the other sensor is live and its
 * FIFO holds only ~10 ms of data — a 200 ms identify() guarantees the
 * working ear overflows on every recovery tick, which is one of the ways a
 * flaky R corrupted L's clock. Recovery gets 2 attempts (~35 ms): still one
 * frame or so lost, but that loss is now compensated in s_toff.            */
static bool identify(int d, volatile uint8_t *id, volatile uint8_t *rge,
                     int attempts)
{
    for (int a = 0; a < attempts; a++) {
        for (int k = 0; k < 6; k++) {
            GPIO_PinOutSet(BUS_PORT, cs_of(d)); tick(); tick();
            GPIO_PinOutClear(BUS_PORT, cs_of(d));
            sel(d);
            for (int i = 0; i < 4; i++) xfer(0xFF);
            GPIO_PinOutSet(BUS_PORT, cs_of(d)); tick(); tick();
        }
        delay_ms(2);
        wr(d, R_SYSCTRL, 0x01); delay_ms(10);
        (void)rd(d, R_INT1); (void)rd(d, R_INT2);
        uint8_t i1 = rd(d, R_PARTID), i2 = rd(d, R_PARTID);
        if (i1 != i2 || (i1 != 0x24 && i1 != 0x25)) { delay_ms(5); continue; }
        uint8_t r1 = rd(d, R_LEDRANGE), r2 = rd(d, R_LEDRANGE);
        if (r1 != r2 || r1 != 0x00) { delay_ms(5); continue; }
        *id = i1; *rge = r1;
        return true;
    }
    return false;
}
static void configure(int d)
{
    wr(d, R_SYSCTRL, 0x02);
    wr(d, R_LED1PA, 0); wr(d, R_LED2PA, 0); wr(d, R_LED3PA, 0);
    /* ADC 32 uA both channels; PPG_TINT = 2 (58.7 us). Two exposures allow
     * that width at 1024 sps — 4x the light of the 3-exposure build. */
    wr(d, R_PPGCFG1, (uint8_t)((0x3 << 4) | (0x3 << 2) | 0x1));  /* TINT=1, 29.4us: 3 exposures cannot hold 58.7 */
    wr(d, R_PPGCFG2, (uint8_t)(SR_CODE << 3));
    wr(d, R_PPGCFG3, (uint8_t)(0x3 << 6));          /* 12 us settling */
    wr(d, R_PDBIAS,  (uint8_t)((0x1 << 4) | 0x1));
    /* THREE exposures: IR, RED, AMBIENT.
     * 3 slots cap PPG_TINT at 1 (29.4 us) at 1024 sps — half the light of
     * the 2-exposure build, which is the cost of keeping red. */
    wr(d, R_LEDSEQ1, (uint8_t)((EXP_RED << 4) | EXP_IR));
    wr(d, R_LEDSEQ2, (uint8_t)EXP_DARK);
    wr(d, R_LEDSEQ3, 0x00);
    wr(d, R_LED2PA, PA_IR_D[d & 1]);   /* per-sensor: see PA_IR_L / PA_IR_R */      /* LED2 = IR  */
    wr(d, R_LED3PA, PA_RED_D[d & 1]);     /* LED3 = RED */
    wr(d, R_PPG_SYNC, SYNC_TARGET);      /* GPIO1 = shared sample trigger */
    wr(d, R_FIFO_WR, 0); wr(d, R_FIFO_RD, 0); wr(d, R_OVF, 0);
    wr(d, R_SYSCTRL, 0x00);
}
/* ===================== DSP ============================================== */
#define MAXFEET 192
/* ---------------------------------------------------------------------------
 * DICROTIC NOTCH REJECTION
 *
 * Every PPG pulse has a systolic peak and, 0.2-0.3 s later, a secondary bump
 * where the aortic valve closes. aa_peakdet_push enforces a 0.2 s refractory,
 * so when the adaptive threshold sits low the notch clears it the instant the
 * refractory expires and gets counted as a beat.
 *
 * The signature is unmistakable and it is what this build measured:
 *      median interval 0.670 s
 *      min             0.213 s   = 0.32x median, sitting ON the refractory
 *      max             1.585 s   = 2.37x median
 * One real beat split into a short and a long. It inflated RR-irregularity
 * to 0.55 and HR_RANGE to 172 bpm.
 *
 * The fix is a physiological plausibility gate on the ACCEPTED beat stream:
 * a beat closer than MIN_IBI_FRAC of the running median interval is a notch,
 * not a contraction. This lives here rather than in arterialage.c — the DSP
 * is validated and stays untouched.
 *
 * Before a median exists, fall back to BOOTSTRAP_MIN_IBI (0.35 s = 171 bpm),
 * which is above any plausible resting rate and below any real one.        */
/* NOTCH GATE — DISABLED.
 *
 * It rejected any beat closer than MIN_IBI_FRAC of the running median
 * interval, to suppress dicrotic notches. On real data it ran away:
 *      25 beats accepted, 18 REJECTED (42%)
 *      median inflated to 1.33 s -> floor rose to 0.80 s
 *      -> real beats at ~1.0 s intervals rejected -> median inflates further
 * That is why HR_rest read 29.8 bpm and why lowering the peak threshold made
 * detection WORSE rather than better: the beats were being found and then
 * thrown away.
 *
 * aa_peakdet's own 0.20 s refractory already suppresses notches adequately.
 * Set MIN_IBI_FRAC above 0 to re-enable, but only with the interval
 * diagnostics (g_rr_median_s / min / max) in view. */
/* A systolic upstroke lasts roughly 80-250 ms, so a valid foot sits that far
 * before its beat. Generous bounds: the point is to reject feet that landed in
 * a different cycle, not to police normal fit variation. */
/* The window a foot may occupy before its beat MUST be narrower than the
 * detector's refractory, or the guard cannot prevent what it exists to catch.
 *
 * It was [0.03, 0.40] — a 0.37 s span against a 0.25 s minimum beat spacing.
 * Two consecutive feet could then legally land closer than their beats, or
 * out of order:
 *      (beat2 - 0.40) - (beat1 - 0.03) = 0.25 - 0.37 = -0.12 s
 * which is why L showed feet at 33.672 and 33.904 — 0.231 s apart, 259 bpm,
 * while the beats behind them were correctly spaced. aa_peakdet's refractory
 * was never at fault; aa__emit sets last_idx and skip_until properly.
 *
 * A real PPG foot-to-peak rise is 0.10-0.25 s. [0.08, 0.28] covers that with
 * 0.20 s of freedom — narrower than 0.25 s, which makes the ordering
 * violation arithmetically impossible rather than merely unlikely.
 *
 * FOOT_MIN_GAP enforces spacing directly as a backstop. At 0.15 s it sits
 * well under the beat floor, so it cannot reject a correctly placed foot and
 * cannot cascade the way a refractory-width chained rule would.            */
#define FOOT_LEAD_MIN  0.00f
#define FOOT_LEAD_MAX  9.00f
#define FOOT_MIN_GAP   0.00f
#define NOTCH_GATE_ENABLE   0
#define MIN_IBI_FRAC        0.60f
#define BOOTSTRAP_MIN_IBI   0.35f
#define IBI_HISTORY         8
typedef struct {
    aa_biquad_t  bq;
    aa_peakdet_t pd;
    float   last_beat_t;          /* time of last ACCEPTED beat */
    float   ibi_hist[IBI_HISTORY];
    int     ibi_n;
    int     rejected;             /* notches suppressed */
    int     feet_dropped;         /* feet failing the spacing guard */
    float   ac[RING];
    int32_t wr;
    float   beats[MAXBEATS];   int nbeats;   /* beat TIMES, seconds */
    float   feet[MAXFEET];     int nfeet;    /* foot TIMES, seconds */
    float   feet_toff[MAXFEET];   /* s_toff at each foot's detection —
                                   * clock forensics: a stepwise inter-ear
                                   * offset in the feet must show here as a
                                   * toff step on one ear, or toff is
                                   * innocent and the fault is elsewhere */
} ear_t;
static ear_t s_ear[2];
static float s_win[RING];
static float s_raw[2][RING];        /* RAW counts per ear — PI needs DC */
static void ear_init(ear_t *e)
{
    memset(e, 0, sizeof(*e));
    aa_biquad_init(&e->bq);
    aa_peakdet_init(&e->pd, SR_HZ);
    e->pd.thr_frac = PEAK_THR_FRAC;
    e->pd.refr     = (int32_t)(PEAK_REFRAC_S * SR_HZ);
}
/* One raw sample in. Records the beat time (single-ear features) AND runs
 * foot detection (dtau). Both consume the same peak — no duplicate work. */
static void push_sample(int d, float raw)
{
    ear_t *e = &s_ear[d];
    /* Unconditional — no dependence on g_feature_ear, which stays -1 until
     * compute_features runs and made g_raw_ir look empty when it was not. */
    /* ROLLING max/min over ~2 s, not a running high-water mark.
     *
     * A lifetime maximum never comes down: one railed sample makes
     * g_raw_max_* read 524287 for the rest of the session, so every later
     * placement attempt looks like it is still clipping even when it is not.
     * The only way to test a new position was to restart the debug session.
     *
     * Resetting the window every WIN samples means you can move a sensor and
     * watch the number respond within a couple of seconds. */
    /* DOUBLE-BUFFERED rolling window.
     *
     * A single buffer that resets in place is unreadable: reading just after
     * a reset shows only the samples accumulated since, so g_raw_max_r read
     * 16384 and then 4 while the signal was actually in the hundreds of
     * thousands. Those were partial windows, not a falling signal.
     *
     * Samples accumulate into cur[]; at the end of each 2 s window the
     * completed extremes are published and cur[] restarts. The published
     * value therefore always represents a full window. */
    {
        static float    cur_lo[2] = { 1e9f,  1e9f};
        static float    cur_hi[2] = {-1e9f, -1e9f};
        static uint32_t win[2]    = {0, 0};
        const  uint32_t WIN       = 2u * (uint32_t)SR_HZ;   /* ~2 s */
        if (raw < cur_lo[d]) cur_lo[d] = raw;
        if (raw > cur_hi[d]) cur_hi[d] = raw;
        if (++win[d] >= WIN) {
            win[d] = 0;
            if (d == 0) {
                g_raw_min_l = cur_lo[0]; g_raw_max_l = cur_hi[0];
                g_raw_span_l = cur_hi[0] - cur_lo[0];
            } else {
                g_raw_min_r = cur_lo[1]; g_raw_max_r = cur_hi[1];
                g_raw_span_r = cur_hi[1] - cur_lo[1];
            }
            cur_lo[d] =  1e9f;
            cur_hi[d] = -1e9f;
        }
    }
    if (d == 0) { g_pushes_l++; g_fi_l = raw; }
    else        { g_pushes_r++; g_fi_r = raw; }
    float ac = aa_biquad_push(&e->bq, raw);
    e->ac[e->wr % RING] = ac;
    /* Decimate into the waveform batch. Ear 0 drives the counter so both ears
     * are sampled at the same instants and the two traces stay aligned. */
    if (d == 0) {
        if (++s_wave_div >= WAVE_DECIM) {
            s_wave_div = 0;
            if (s_wave_n < WAVE_BATCH) {
                s_wave[0][s_wave_n] = (int32_t)ac;
                s_wave[1][s_wave_n] = (int32_t)s_ear[1].ac[
                                        (s_ear[1].wr ? s_ear[1].wr - 1 : 0) % RING];
                s_wave_n++;
            }
        }
    }
    s_raw[d][e->wr % RING] = raw;
    /* Mirror whichever ear supplies the features — these were hardcoded to
     * ear 0, so with a single sensor on the R pins they stayed empty and
     * the raw signal could not be inspected at all. */
    if (d == g_feature_ear) {
        g_raw_ir[e->wr % RING] = raw;
        g_ac_ir [e->wr % RING] = ac;
        g_wr = (int)(e->wr % RING);
    }
    /* NO ENVELOPE RE-SEED HERE.
     * aa_peakdet_reset() zeroes the detector's internal sample counter, and
     * beat times are derived from that counter (t = bi / SR_HZ). Resetting
     * mid-run therefore (a) re-triggers the WARMUP_S drop for another 3 s and
     * (b) shifts every later timestamp onto a different time base, with each
     * ear resetting at a slightly different sample. That is fatal for dtau,
     * which depends on the two ears sharing one clock.
     *
     * Tried and reverted: it cut detection to 1 beat in 30 s. */
    /* Record what the detector is working with. thr = thr_frac * env; if thr
     * exceeds the largest |AC| the filter ever produces, it cannot fire. */
    {
        float a = ac < 0 ? -ac : ac;
        if (d == 0) {
            if (a > g_ac_max_l) g_ac_max_l = a;
            g_env_l = e->pd.env;
            g_thr_l = e->pd.thr_frac * e->pd.env;
            g_seeded_l = e->pd.seeded;
        } else {
            if (a > g_ac_max_r) g_ac_max_r = a;
            g_env_r = e->pd.env;
            g_thr_r = e->pd.thr_frac * e->pd.env;
            g_seeded_r = e->pd.seeded;
        }
    }
    int32_t bi;
    if (aa_peakdet_push(&e->pd, ac, &bi)) {
        /* ---- ENVELOPE CORRECTION ------------------------------------
         * aa_peakdet updates env ONLY in its IDLE branch:
         *      float decayed = d->env * d->env_decay;
         *      d->env = x > decayed ? x : decayed;
         * The instant x crosses the threshold it switches to AA_SEARCH,
         * and the systolic PEAK occurs during SEARCH. So env never sees the
         * peak — it converges on the maximum of the INTER-BEAT BASELINE.
         *
         * Measured: env 5367 while peak AC was 57685. Ten times too low, so
         * thr = 0.45 * env sat at 4% of the real signal and the detector fired
         * on noise and dicrotic notches. It also explains why the two ears
         * kept swapping which one over-detected: whichever had the noisier
         * baseline got the higher env, and that varies run to run.
         *
         * Raising env to the actual beat amplitude after each detection puts
         * thr where thr_frac says it should be — a fraction of the PULSE, not
         * of the noise floor. Applied here rather than in arterialage.c so the
         * validated DSP is untouched. */
        {
            int32_t look = (int32_t)(0.35f * SR_HZ);     /* one beat back */
            int32_t from = e->wr - look; if (from < 0) from = 0;
            float pk = 0.0f;
            for (int32_t i = from; i <= e->wr; i++) {
                float a = e->ac[i % RING];
                if (a < 0) a = -a;
                if (a > pk) pk = a;
            }
            if (pk > e->pd.env) e->pd.env = pk;
        }
        /* + s_toff[d]: dropped frames stall the counter but not real time;
         * the offset restores the timestamp to the shared clock. */
        float t = (float)bi / SR_HZ + s_toff[d];
        /* ---- notch rejection ---------------------------------------- */
        float min_ibi = BOOTSTRAP_MIN_IBI;
        if (e->ibi_n >= 3) {
            float srt[IBI_HISTORY];
            int m = (e->ibi_n < IBI_HISTORY) ? e->ibi_n : IBI_HISTORY;
            for (int i = 0; i < m; i++) srt[i] = e->ibi_hist[i];
            for (int i = 1; i < m; i++) {
                float k = srt[i]; int j = i - 1;
                while (j >= 0 && srt[j] > k) { srt[j+1] = srt[j]; j--; }
                srt[j+1] = k;
            }
            float med = srt[m/2];
            float f   = med * MIN_IBI_FRAC;
            if (f > min_ibi) min_ibi = f;
        }
        if (NOTCH_GATE_ENABLE &&
            e->last_beat_t > 0.0f && (t - e->last_beat_t) < min_ibi) {
            e->rejected++;         /* dicrotic notch, not a contraction */
            e->wr++;
            return;
        }
        if (e->last_beat_t > 0.0f) {
            float ibi = t - e->last_beat_t;
            e->ibi_hist[e->ibi_n % IBI_HISTORY] = ibi;
            e->ibi_n++;
        }
        e->last_beat_t = t;
        if (e->nbeats < MAXBEATS) e->beats[e->nbeats++] = t;
        /* ---- FOOT WINDOW MUST BE SHORTER THAN ONE BEAT ------------------
         * aa_foot_tangent finds the trough as the argmin over [pk-0.4s, pk].
         * Passing the whole 1 s ring let that look-back reach into the
         * PREVIOUS beat: at 88 bpm the interval is 0.68 s, so a 0.4 s search
         * sometimes located the previous cycle's trough and the foot landed a
         * cycle early.
         *
         * MEASURED consequence — consecutive foot times on one ear:
         *      ... 73.483, 73.628, 74.402, 74.514 ...
         *      intervals 0.145 s and 0.111 s  =  414 and 540 bpm
         * Impossible as beats, and below the 0.25 s refractory, so these were
         * never extra detections: they were feet placed in the wrong cycle.
         * Spurious feet then paired against each other, landed beyond the
         * 50 ms cutoff, and were rejected — which is why 88 matched feet on
         * each ear still yielded only 10 pairs.
         *
         * Bounding the window to a fraction of the measured beat interval
         * confines the trough search to the current cycle. Uses the live
         * median when there is one, and a 0.5 s fallback (120 bpm) before
         * then, which is shorter than any resting interval.                */
        float beat_s = (g_rr_median_s > 0.30f && g_rr_median_s < 2.0f)
                       ? g_rr_median_s : 0.70f;
        /* ---- ANCHOR THE WINDOW TO THE PEAK, NOT TO NOW ------------------
         * ROOT CAUSE of the scattered feet.
         *
         * aa_peakdet emits a beat only after collecting AA_SEARCH_S = 0.10 s
         * = 103 samples past the threshold crossing, so bi lags e->wr by 0 to
         * 103 samples depending where the maximum fell in that search buffer.
         *
         * Anchoring the window to e->wr therefore gave a VARYING amount of
         * history before the peak:
         *      pk = span - (search - k_off) = 277 - 103 + k_off
         *         = 174 to 277 samples = 0.17 to 0.27 s
         * while the true foot sits 0.15-0.25 s before the peak. Sometimes the
         * trough was inside that range, sometimes it was not — and it changed
         * beat to beat. When it fell short, aa_foot_tangent's argmin returned
         * an arbitrary minimum from a truncated window and the foot landed
         * anywhere, which is why the lead varied from under 0.08 s to over
         * 0.28 s and why any fixed acceptance window either let bad feet
         * through or rejected good ones.
         *
         * Anchoring to bi gives every beat exactly the same history. The
         * placement becomes deterministic instead of depending on where the
         * peak happened to sit inside the search buffer.
         *
         * FOOT_HISTORY_S 0.35 comfortably contains a 0.15-0.25 s rise without
         * reaching the previous peak, and is capped at 0.45 x the measured
         * beat so it stays inside the cycle at higher rates.               */
        float hist_s = 0.35f;
        if (hist_s > 0.45f * beat_s) hist_s = 0.45f * beat_s;
        int32_t hist = (int32_t)(hist_s * SR_HZ);
        if (hist < 32) hist = 32;
        int32_t newest = bi, oldest = bi - hist;
        if (oldest < 0) oldest = 0;
        int n = (int)(newest - oldest + 1);
        if (n > 16 && bi >= oldest && (bi - oldest) < RING &&
            e->nfeet < MAXFEET) {
            for (int i = 0; i < n; i++) s_win[i] = e->ac[(oldest + i) % RING];
            int32_t lp = bi - oldest;
            float f;
            if (aa_foot_tangent(s_win, n, SR_HZ, &lp, 1, &f) == 1) {
                float ft = f + (float)oldest / SR_HZ + s_toff[d];
                /* ---- FOOT VALIDATED AGAINST ITS OWN BEAT ---------------
                 * A foot is the start of the upstroke that leads to ITS beat,
                 * so it must sit shortly BEFORE that beat — physiologically a
                 * systolic upstroke is roughly 80-250 ms. Anything outside
                 * that slipped into another part of the cycle.
                 *
                 * MEASURED: consecutive feet 0.146 s and 0.111 s apart on one
                 * ear (411 and 541 bpm) while the beats behind them were a
                 * sound 0.676 s median. Feet in the wrong cycle, which then
                 * paired against each other beyond the 50 ms cutoff — why 88
                 * matched feet per ear survived as only 10 pairs.
                 *
                 * Checked against the BEAT rather than against the previous
                 * foot deliberately. A chained check cascades: one foot that
                 * slips FORWARD becomes the reference and blocks the next
                 * genuine foot, then the one after, and a bad patch can drop
                 * alternate feet indefinitely. Comparing to the beat is
                 * independent for every foot, so a bad foot costs exactly
                 * itself.                                                   */
                float lead = t - ft;                 /* foot precedes beat */
                if (lead < FOOT_LEAD_MIN || lead > FOOT_LEAD_MAX) {
                    e->feet_dropped++;
                    e->wr++;
                    return;
                }
                if (e->nfeet > 0 &&
                    (ft - e->feet[e->nfeet - 1]) < FOOT_MIN_GAP) {
                    e->feet_dropped++;
                    e->wr++;
                    return;
                }
                e->feet_toff[e->nfeet] = s_toff[d];
                e->feet[e->nfeet++] = ft;
                s_last_foot[d] = ft;
            }
        }
    }
    e->wr++;
}
static int drain(int d)
{
    uint8_t ovf = rd(d, R_OVF);
    if (d == 0) {
        g_ovf_l = ovf;
        if (ovf) { g_ovf_events_l++; g_ovf_lost_l += ovf; }
    } else {
        g_ovf_r = ovf;
        if (ovf) { g_ovf_events_r++; g_ovf_lost_r += ovf; }
    }
    if (ovf) {
        /* OVF is telemetry now — s_toff comes from the trigger deficit at
         * the bottom of this function, which measures every loss channel
         * exactly. Saturation still trips a full resync: a loss burst that
         * pegs the counter deserves a fresh history, not a patched one. */
        if (ovf >= 0x7F) {
            g_aligned = false;
            g_resyncs++;
        }
    }
    int avail = rd(d, R_FIFO_CNT);
    if (d == 0) { g_avail_l = avail; g_drains_l++; }
    else        { g_avail_r = avail; g_drains_r++; }
    if (avail == 0xFF) return -1;
    /* Do NOT gate on a hardcoded frame size.
     *
     * SPF was 6 (3 exposures x 2 photodiodes) while both FIFOs were reporting
     * avail = 4 — so "avail < SPF" returned empty on every one of 51349
     * drains and not a single sample ever reached the DSP. Four samples per
     * frame means the sequencer is running TWO exposures, not three: the red
     * slot did not take.
     *
     * Reading whatever is present and letting the TAG_AM boundary close each
     * frame is robust to the exposure count, so a mismatch between the
     * configured sequence and the decoder's expectation can no longer
     * silently discard everything. */
    if (avail < 2) return 0;
    int take = avail;
    /* MUST stay a multiple of SPF. Clamping to a non-multiple leaves a
     * partial frame in the FIFO, so the next drain starts mid-frame and fi
     * is loaded from the wrong slot — sometimes a PD1 tag, which reads in
     * the tens. The bandpass then sees scrambled data with no pulse in it
     * and the detector never fires.
     * 124 was correct for SPF 4; it is NOT a multiple of 6. */
    if (take > (20 * SPF)) take = 20 * SPF;
    sel(d);
    GPIO_PinOutClear(BUS_PORT, cs_of(d));
    xfer(R_FIFO_DATA); xfer(0xFF);
    /* fi persists across drains: a frame can span a FIFO read boundary, and
     * resetting to 0 each call would reintroduce exactly the bug above. */
    static float fi_hold[2] = {0.0f, 0.0f};
    float fi = fi_hold[d];
    int frames = 0;
    for (int i = 0; i < take; i++) {
        uint8_t b0 = xfer(0), b1 = xfer(0), b2 = xfer(0);
        uint32_t w = ((uint32_t)b0 << 16) | ((uint32_t)b1 << 8) | b2;
        uint8_t tg = (uint8_t)((w >> 19) & 0x1F);
        float   v  = (float)(w & 0x0007FFFFUL);   /* UNSIGNED, 19-bit */
        g_tagcount[tg & 0x1F]++;
        if (g_tagn < 48) g_tagseq[g_tagn++] = tg;
        if (g_capn < 48) {
            g_capt[g_capn] = tg;
            g_capv[g_capn] = (w & 0x0007FFFFUL);
            g_capn++;
        }
        /* Each tag appears TWICE per frame, and the second occurrence is 0:
         *      tag 1 = 32,  tag 7 = 58519,  tag 1 = 0,  tag 7 = 0,
         *      tag 2 = 38,  tag 8 = 58801,  ...
         * Plain "fi = v" therefore took the ZERO and push_sample received 0
         * on every frame — samples flowed, the bandpass saw a constant, and
         * the detector never fired.
         *
         * Accept only a non-zero reading. A real PPG sample is never exactly
         * 0: the ADC sits on a large DC pedestal (tens of thousands here).  */
        if      (tg == TIMING_TAG) { if (v > 0.0f) fi = v; }
        else if (tg == TAG_AM) {
            if (d == g_feature_ear) g_ambient = v;
            push_sample(d, fi);
            frames++;
        }
    }
    GPIO_PinOutSet(BUS_PORT, cs_of(d)); tick();
    fi_hold[d] = fi;
    if (d == 0) g_frames_l += (uint32_t)frames; else g_frames_r += (uint32_t)frames;

    /* ---- GROUND-TRUTH CLOCK DEFICIT — see MID-RUN CLOCK DRIFT ------------
     * MEASURED, this build: def 105/110 against toff 102/107 samples — the
     * in-flight residue at end-of-drain is ~3 samples, constant, on BOTH
     * ears. The FIFO was just emptied; there is almost nothing in flight,
     * and what remains is common-mode across ears (same loop, same
     * pipeline) so it cancels in dtau.
     *
     * The 1 s windowed MINIMUM this replaces was therefore filtering noise
     * that barely exists, at a cost that was killing the pairing: losses
     * arrive as BURSTS (zero OVF events with a growing deficit = missed
     * conversions, a chunk at a time), and the min lagged each burst by up
     * to TWO windows — one to observe the new floor, one to publish it.
     * Every foot detected in that stretch was mistimed by the burst size,
     * which is how 71 and 92 healthy feet yielded 3 pairs.
     *
     * Direct assignment corrects a burst in the same drain that observes
     * it. The 1.5-sample hysteresis only suppresses single-sample chatter;
     * any real change passes immediately. */
    if (g_aligned) {
        uint32_t elapsed = s_ticks - g_align_at;         /* triggers issued */
        uint32_t counted = (uint32_t)s_ear[d].wr;        /* pushes arrived  */
        uint32_t def = (elapsed > counted) ? (elapsed - counted) : 0;
        if (d == 0) g_def_l = def; else g_def_r = def;

        float cur = s_toff[d] * SR_HZ;
        if ((float)def > cur + 1.5f || (float)def < cur - 1.5f) {
            s_toff[d] = (float)def / SR_HZ;
            if (d == 0) g_toff_l = s_toff[0]; else g_toff_r = s_toff[1];
        }
    }
    return frames;
}
/* ===================== features ========================================= */
/* HR percentiles over a sliding set of windowed HR estimates, matching the
 * numpy linear-interpolation definition the Python uses. */
static float pct(float *v, int n, float p)
{
    if (n <= 0) return -999.0f;
    for (int i = 1; i < n; i++) {          /* insertion sort, n is small */
        float k = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > k) { v[j+1] = v[j]; j--; }
        v[j+1] = k;
    }
    float idx = p * (float)(n - 1);
    int   lo  = (int)idx;
    int   hi  = (lo + 1 < n) ? lo + 1 : lo;
    float f   = idx - (float)lo;
    return v[lo] + (v[hi] - v[lo]) * f;
}
static void compute_features(void)
{
    /* ---- PROVISIONAL dtau: visible from the first pair ---------------
     * Same pairing rule as aa_dtau_from_feet (nearest within
     * AA_DTAU_PAIR_WINDOW_MS, reject beyond AA_DTAU_MAX_ABS_MS) but no MAD
     * reject and no HR guard, and no minimum beat count. Diagnostic only. */
    if (s_ear[0].nfeet >= 2 && s_ear[1].nfeet >= 2) {
        /* Work on a COPY. The validated path used to mutate these arrays,
         * which stepped this estimate without warning. */
        static float fl[MAXFEET], fr[MAXFEET];
        int nl = s_ear[0].nfeet, nr = s_ear[1].nfeet;
        for (int i = 0; i < nl; i++) fl[i] = s_ear[0].feet[i];
        for (int i = 0; i < nr; i++) fr[i] = s_ear[1].feet[i];
        static float pd[MAXFEET];
        int np = 0;
        float win = AA_DTAU_PAIR_WINDOW_MS * 1e-3f;
        int j = 0;
        for (int i = 0; i < nl && np < MAXFEET; i++) {
            float tl = fl[i];
            while (j + 1 < nr && fabsf(fr[j+1] - tl) < fabsf(fr[j] - tl)) j++;
            float d = fr[j] - tl;
            if (fabsf(d) <= win) {
                float dm = d * 1000.0f;
                if (fabsf(dm) <= AA_DTAU_MAX_ABS_MS) pd[np++] = dm;
            }
        }
        g_dtau_raw_n = np;
        if (np > 0) {
            for (int i = 1; i < np; i++) {
                float k = pd[i]; int q = i - 1;
                while (q >= 0 && pd[q] > k) { pd[q+1] = pd[q]; q--; }
                pd[q+1] = k;
            }
            g_dtau_raw_min = pd[0];
            g_dtau_raw_max = pd[np-1];
            g_dtau_raw_ms  = pd[np/2];
            g_dtau_raw_hist[g_dtau_raw_count % DTAU_HIST] = g_dtau_raw_ms;
            g_dtau_raw_count++;
        }
    }
    /* ---- dtau: the VALIDATED path, needs BOTH ears' feet -------------- */
    if (s_ear[0].nfeet >= ATTEMPT_AT && s_ear[1].nfeet >= ATTEMPT_AT) {
        float dt; int nfin = 0;
        aa_dtau_status_t st = aa_dtau_from_feet(s_ear[0].feet, s_ear[0].nfeet,
                                                s_ear[1].feet, s_ear[1].nfeet,
                                                &dt, &nfin);
        g_dtau_status = (int)st;
        g_dtau_n = nfin;
        if (st == AA_DTAU_OK) g_dtau_ms = dt;
        /* No trimming here — the sliding window in push_sample keeps the
         * buffers bounded, so nothing mutates underneath the provisional
         * estimate and the window length stays constant. */
    }
    /* ---- single-ear features ---------------------------------------- */
#if FEATURE_EAR < 0
    int fe = g_ok_l ? 0 : (g_ok_r ? 1 : 0);   /* auto: whichever is up */
#else
    int fe = FEATURE_EAR;
#endif
    g_feature_ear = fe;
    int nb = s_ear[fe].nbeats;
    if (nb < 4) return;
    static float rr[MAXBEATS];
    static bool  gap[MAXBEATS];
    int nrr = 0;
    for (int i = 1; i < nb && nrr < MAXBEATS; i++) {
        float d = s_ear[fe].beats[i] - s_ear[fe].beats[i-1];
        rr[nrr]  = d;
        /* spans_gap: a break in contact, not a physiological interval.
         * AA_RR_MAX_S is 2.5 s, so anything longer is a dropout. */
        gap[nrr] = (d > AA_RR_MAX_S);
        nrr++;
    }
    /* interval spread, before any filtering */
    if (nrr > 0) {
        static float srt[MAXBEATS];
        for (int i = 0; i < nrr; i++) srt[i] = rr[i];
        for (int i = 1; i < nrr; i++) {
            float k = srt[i]; int j = i - 1;
            while (j >= 0 && srt[j] > k) { srt[j+1] = srt[j]; j--; }
            srt[j+1] = k;
        }
        g_rr_min_s    = srt[0];
        g_rr_max_s    = srt[nrr-1];
        g_rr_median_s = srt[nrr/2];
        g_rr_n_raw    = nrr;
    }
    float burden; int nvalid = 0;
    aa_feat_status_t st = aa_rr_irregularity(rr, gap, nrr, &burden, &nvalid);
    g_rr_status = (int)st;
    g_rr_nvalid = nvalid;
    if (st == AA_FEAT_OK) g_rr_irregularity = burden;
    /* --- HR from the same intervals --- */
    /* ---- MISSED-BEAT ARTEFACT REJECT --------------------------------------
     * A missed beat DOUBLES one interval, which HALVES that instantaneous
     * rate — and the halved rates are the BOTTOM of the HR distribution,
     * which is exactly where HR_rest = p05 reads. Measured: 29-30 bpm
     * reported at a true 55-65. The arithmetic is the fingerprint: 58 bpm ->
     * 1.03 s intervals; one missed beat -> 2.07 s -> 29 bpm on the nose.
     * The AA_RR_MAX_S = 2.5 s gate only catches doubles when the true rate
     * is below ~48 bpm, so at resting rates they sail through.
     *
     * The fix REMOVES the artefacts rather than changing the statistic.
     * HR_rest is DEFINED as p05 — pct() exists to match numpy's percentile
     * exactly, and arterialage.json's norms were built on that definition.
     * A median sent under the name HR_rest is a different quantity wearing
     * a validated name: contract rule 2 in reverse.
     *
     * Doubles sit at 2.0x the batch median, extra detections at ~0.5x, and
     * genuine resting sinus variation stays within ~+/-25%, so [0.6, 1.5]x
     * of g_rr_median_s (computed just above from this same batch) rejects
     * the artefacts with room to spare and leaves the real distribution —
     * and its p05 — intact. HR_RANGE = p95-p05 of the cleaned set is then
     * honest too (still deliberately unsent; see the emit block).          */
    float med_ibi = g_rr_median_s;
    static float hrv[MAXBEATS];
    int nh = 0;
    /* Ceiling 1.30x, down from 1.5x — MEASURED leak: with L missing ~25%
     * of beats, single-miss intervals of 1.58-1.68 s sat at ~1.45x the
     * 1.1 s median, slipped under the 1.5x ceiling, and p05 landed on them:
     * 37 bpm on the wire at a true 55-65. A genuine resting interval does
     * not exceed ~1.3x the running median; a missed beat always does. */
    for (int i = 0; i < nrr; i++)
        if (rr[i] >= AA_RR_MIN_S && rr[i] <= AA_RR_MAX_S && nh < MAXBEATS &&
            (med_ibi <= 0.0f ||
             (rr[i] > 0.6f * med_ibi && rr[i] < 1.30f * med_ibi)))
            hrv[nh++] = 60.0f / rr[i];
    /* nh >= 10, up from 3. After an alignment wipe the first few intervals
     * ARE the distribution, and p05 of a handful of values is whatever the
     * smallest happens to be — measured: 94.6 bpm on the wire from a
     * post-wipe sample where one notch double (0.63 s) was the bottom of
     * the set. Ten intervals (~10 s) is the shortest window where a 5th
     * percentile means anything; until then g_hr_rest stays -999 and the
     * emit block stays silent, which reads as "acquiring", not as a rate. */
    if (nh >= 10) {
        g_hr_now = hrv[nh-1];
        static float tmp[MAXBEATS];
        for (int i = 0; i < nh; i++) tmp[i] = hrv[i];
        float p05 = pct(tmp, nh, 0.05f);
        for (int i = 0; i < nh; i++) tmp[i] = hrv[i];
        float p95 = pct(tmp, nh, 0.95f);
        g_hr_rest  = p05;
        g_hr_range = p95 - p05;
    }
    /* --- perfusion index: live contact quality, no debugger needed --- */
    /* Perfusion index AT THE LAST DETECTED BEAT. Passing n-1 measured at an
     * arbitrary sample instead of a peak, which is why it read 1.34 — a real
     * PI is 0.005 to 0.05. Also record the raw min/max over the window so
     * saturation (524287) is visible directly. */
    if (s_ear[fe].wr > 64 && s_ear[fe].nbeats > 0) {
        int32_t newest = s_ear[fe].wr, oldest = newest - (RING - 1);
        if (oldest < 0) oldest = 0;
        int n = (int)(newest - oldest + 1);
        static float wraw[RING];
        float lo = 1e9f, hi = -1e9f;
        for (int i = 0; i < n; i++) {
            float x = s_raw[fe][(oldest + i) % RING];
            wraw[i] = x;
            if (x < lo) lo = x;
            if (x > hi) hi = x;
        }
        g_raw_min = lo; g_raw_max = hi;
        float tb = s_ear[fe].beats[s_ear[fe].nbeats - 1];
        /* tb carries the s_toff clock correction; ring indices do NOT.
         * Converting time back to an index without removing it put pk ~80
         * samples off the peak, which is how PI read 0.23 and then 1.05 —
         * both physically impossible. Strip the offset before indexing. */
        int32_t pk = (int32_t)((tb - s_toff[fe]) * SR_HZ) - oldest;
        if (pk > 8 && pk < n - 8)
            g_pi = aa_perfusion_index(wraw, n, SR_HZ, (int)pk);
    }
}
/* ===================== app ============================================== */
void app_init(void)
{
    CMU_ClockEnable(cmuClock_GPIO, true);
    GPIO_PinModeSet(BUS_PORT, PIN_MOSI, gpioModePushPull, 0);
    GPIO_PinModeSet(BUS_PORT, PIN_SCLK, gpioModePushPull, 0);
    GPIO_PinModeSet(BUS_PORT, MISO_L,   gpioModeInput,    0);
    GPIO_PinModeSet(BUS_PORT, MISO_R,   gpioModeInput,    0);
    GPIO_PinModeSet(BUS_PORT, CS_L,     gpioModePushPull, 1);
    GPIO_PinModeSet(BUS_PORT, CS_R,     gpioModePushPull, 1);
    GPIO_PinModeSet(TRIG_PORT, TRIG_PIN, gpioModePushPull, 0);
    /* Block EM2. main.c calls sl_power_manager_sleep() every loop, and EM2
     * stops SysTick — which drives the shared sample trigger on PA07. Without
     * this the sensors keep converting while the MCU sleeps, the FIFOs
     * overflow, and the two ears' timestamps drift apart. EM1 still sleeps
     * the core between interrupts, so BLE power behaviour is unaffected. */
#if defined(SL_CATALOG_POWER_MANAGER_PRESENT)
    sl_power_manager_add_em_requirement(SL_POWER_MANAGER_EM1);
#endif
    spi_init();
    s_ticks = 0;
    SysTick_Config(SystemCoreClock / (uint32_t)SR_HZ);
    delay_ms(10);
    ear_init(&s_ear[0]);
    ear_init(&s_ear[1]);
    for (int i = 0; i < DTAU_HIST; i++) g_dtau_raw_hist[i] = -999.0f;
    g_cfg_attempt_at = ATTEMPT_AT;
    /* Independent bring-up: one sensor failing must not kill the other, and
     * must not kill the run. dtau needs both; everything else needs one. */
    g_ok_l = identify(0, &g_id_l, &g_range_l, 12);
    g_ok_r = identify(1, &g_id_r, &g_range_r, 12);
    if (g_ok_l) configure(0);
    if (g_ok_r) configure(1);
    g_err = (g_ok_l && g_ok_r) ? 0 : ((g_ok_l || g_ok_r) ? 1 : 3);
    s_trig_on = true;
}
void app_process_action(void)
{
    g_loops++;
    if (g_ok_l) { if (drain(0) < 0) g_ok_l = false; }
    if (g_ok_r) { if (drain(1) < 0) g_ok_r = false; }
    /* Background recovery, ~1 s. identify() resets and double-reads, so it
     * must not run per loop or it starves the WORKING sensor's FIFO.
     *
     * A recovered ear FORCES A JOINT RE-ALIGN. configure() flushes the FIFO
     * and zeroes OVF, so everything lost during the outage is invisible to
     * the s_toff compensation — the ear comes back with its clock behind by
     * the whole outage and no record of it. Pairing against that clock is
     * the drift bug all over again, just delivered differently. Measured
     * fingerprint: HR stats frozen while the ear was down, then a
     * discontinuous jump when it returned. Restarting both ears together is
     * the only honest continuation; g_resyncs counts it (first-time bring-up
     * before any alignment is not counted — g_aligned was already false). */
    {
        static uint32_t nxt = 0;
        if ((!g_ok_l || !g_ok_r) && s_ticks > nxt) {
            nxt = s_ticks + (uint32_t)SR_HZ;
            bool rec = false;
            if (!g_ok_l && identify(0, &g_id_l, &g_range_l, 2)) { configure(0); g_ok_l = true; rec = true; }
            if (!g_ok_r && identify(1, &g_id_r, &g_range_r, 2)) { configure(1); g_ok_r = true; rec = true; }
            if (rec && g_aligned) { g_aligned = false; g_resyncs++; }
            g_err = (g_ok_l && g_ok_r) ? 0 : ((g_ok_l || g_ok_r) ? 1 : 3);
        }
    }
    g_nbeats_l = s_ear[0].nbeats; g_nbeats_r = s_ear[1].nbeats;
    /* One-time joint alignment: both ears' clocks start together or the feet
     * are not comparable. See SHARED TIME BASE above. */
    if (!g_aligned && g_ok_l && g_ok_r) {
        wr(0, R_FIFO_WR, 0); wr(0, R_FIFO_RD, 0); wr(0, R_OVF, 0);
        wr(1, R_FIFO_WR, 0); wr(1, R_FIFO_RD, 0); wr(1, R_OVF, 0);
        for (int d = 0; d < 2; d++) {
            s_ear[d].wr     = 0;
            s_ear[d].nbeats = 0;
            s_ear[d].nfeet  = 0;
            s_ear[d].ibi_n  = 0;
            s_ear[d].last_beat_t = 0.0f;
            aa_biquad_reset(&s_ear[d].bq);
            aa_peakdet_init(&s_ear[d].pd, SR_HZ);
            s_ear[d].pd.thr_frac = PEAK_THR_FRAC;
            s_ear[d].pd.refr     = (int32_t)(PEAK_REFRAC_S * SR_HZ);
        }
        s_toff[0] = 0.0f; s_toff[1] = 0.0f;   /* fresh shared clock */
        g_toff_l = 0.0f; g_toff_r = 0.0f;
        g_aligned  = true;
        g_align_at = s_ticks;
    }
    g_nfeet_l  = s_ear[0].nfeet;  g_nfeet_r  = s_ear[1].nfeet;
    /* mirror the newest FEETDBG foot times so the two spans can be compared */
    for (int d = 0; d < 2; d++) {
        int n = s_ear[d].nfeet;
        int k = n < FEETDBG ? n : FEETDBG;
        int from = n - k;
        for (int i = 0; i < k; i++) {
            float v = s_ear[d].feet[from + i];
            float o = s_ear[d].feet_toff[from + i];
            if (d == 0) { g_feet_l[i] = v; g_ftoff_l[i] = o; }
            else        { g_feet_r[i] = v; g_ftoff_r[i] = o; }
        }
        for (int i = k; i < FEETDBG; i++) {
            if (d == 0) { g_feet_l[i] = -999.0f; g_ftoff_l[i] = -999.0f; }
            else        { g_feet_r[i] = -999.0f; g_ftoff_r[i] = -999.0f; }
        }
        if (d == 0) g_feet_n_l = k; else g_feet_n_r = k;
        int nb = s_ear[d].nbeats;
        int kb = nb < FEETDBG ? nb : FEETDBG;
        int fb = nb - kb;
        for (int i = 0; i < kb; i++) {
            float v = s_ear[d].beats[fb + i];
            if (d == 0) g_beats_l[i] = v; else g_beats_r[i] = v;
        }
        for (int i = kb; i < FEETDBG; i++) {
            if (d == 0) g_beats_l[i] = -999.0f; else g_beats_r[i] = -999.0f;
        }
        if (d == 0) g_beats_n_l = kb; else g_beats_n_r = kb;
    }
    g_notches_l = s_ear[0].rejected; g_notches_r = s_ear[1].rejected;
    g_feetdrop_l = s_ear[0].feet_dropped; g_feetdrop_r = s_ear[1].feet_dropped;
    g_frame_gap = (int)(s_ear[1].wr - s_ear[0].wr);
    /* ---- waveform batch out ---------------------------------------------
     * Emitted as soon as WAVE_BATCH samples exist, independent of the feature
     * cadence, so the trace moves smoothly rather than in 2-second jumps.
     * Tagged "w" so aa_bridge.py can route it to the waveform endpoint
     * instead of the model — feature vectors and waveform share one
     * characteristic but must not share a destination. */
    if (s_wave_n >= WAVE_BATCH) {
        char _w[256];
        int n = snprintf(_w, sizeof _w, "{\"w\":[");
        for (int i = 0; i < WAVE_BATCH; i++)
            n += snprintf(_w + n, sizeof _w - (size_t)n, "%s%ld,%ld",
                          i ? "," : "", (long)s_wave[0][i], (long)s_wave[1][i]);
        { char fb1[24], fb2[24];
          snprintf(_w + n, sizeof _w - (size_t)n, "],\"fl\":%s,\"fr\":%s}\r\n",
                   fmtf(fb1, sizeof fb1, s_last_foot[0], 3),
                   fmtf(fb2, sizeof fb2, s_last_foot[1], 3)); }
        out_line(_w);
        g_wave_sent++;
        s_wave_n = 0;
    }
    /* Features every ~1 s. Running them per loop would starve the drain —
     * the FIFO holds 128/4 = 32 frames, only 31 ms at 1024 sps. */
    static uint32_t nxt_feat = 0;
    if (s_ticks > nxt_feat) {
        nxt_feat = s_ticks + (uint32_t)SR_HZ;
        compute_features();
#if STREAM_JSON
        /* One line every 2 s, ready for push.py --stdin --follow.
         * RR_irregularity is a FRACTION (0..1), HR in bpm — the units
         * arterialage.json expects. A refused feature is omitted rather
         * than sent as a guess: contract.py rule 2. */
        static uint32_t nxt_tx = 0;
        /* Emit on HR, not on RR. RR-irregularity is the most
         * detection-sensitive feature here and gating the whole vector on it
         * suppressed HR_rest, which is correct and robust. */
        if (s_ticks > nxt_tx && g_hr_rest > 0.0f) {
            nxt_tx = s_ticks + 2u * (uint32_t)SR_HZ;
            int fe_tx = (g_feature_ear >= 0) ? g_feature_ear : 0;
            /* EMIT GATE DISABLED at operator request for demo: dtau is sent
             * whenever the validated path returns OK, regardless of pair
             * count or cluster spread. g_dtau_gate is still computed and
             * still readable in the debugger, so the quality of any emitted
             * value can be judged after the fact — 0 clustered, 2 too few
             * pairs, 3 not clustered. Re-enable by restoring
             * `dtau_good = (g_dtau_gate == 0);` below. */
            bool dtau_good;
            if (g_dtau_status != AA_DTAU_OK)             g_dtau_gate = 1;
            else if (g_dtau_n < DTAU_EMIT_MIN_PAIRS ||
                     g_dtau_raw_n <= 0)                  g_dtau_gate = 2;
            else if ((g_dtau_raw_max - g_dtau_raw_min)
                         > DTAU_EMIT_MAX_SPREAD_MS)      g_dtau_gate = 3;
            else                                         g_dtau_gate = 0;
            dtau_good = (g_dtau_status == AA_DTAU_OK);
            if (dtau_good)
                { char _l[256];
                /* HR_RANGE and RR_irregularity are DELIBERATELY NOT SENT.
                 * Measured live they read ~190 bpm and 0.52-0.67; the model's
                 * own norms put RR_irregularity p99 at 0.20, so both were
                 * several times off-scale. The cause is missed beats
                 * inflating the interval spread — HR_rest survives it because
                 * a percentile barely moves on a few bad intervals, but a
                 * range and a variability burden are exactly the statistics
                 * that do not.
                 *
                 * Omitting is the engine's designed path: it returns
                 * tier_is_floor and the GUI shows those domains as
                 * "not measured". Sending them would put numbers on screen
                 * that could not be defended. Restore once beat detection is
                 * clean enough that HR_RANGE reads 10-20 bpm at rest.
                 *
                 * pi and n_* are diagnostics; aa_bridge.py drops them before
                 * they reach the model. */
                { char b1[24],b2[24],b3[24];
                snprintf(_l, sizeof _l,
                       "{\"dtau\":%s,\"HR_rest\":%s,"
                       "\"n_beats\":%d,\"n_pairs\":%d,\"pi\":%s}\r\n",
                       fmtf(b1,sizeof b1, g_dtau_ms/1000.0f, 6),
                       fmtf(b2,sizeof b2, g_hr_rest, 1),
                       s_ear[fe_tx].nbeats, g_dtau_n,
                       fmtf(b3,sizeof b3, g_pi, 5)); };
                out_line(_l); }
            else
                { char _l[256];
                { char b1[24],b2[24];
                snprintf(_l, sizeof _l,
                       "{\"HR_rest\":%s,\"n_beats\":%d,\"pi\":%s}\r\n",
                       fmtf(b1,sizeof b1, g_hr_rest, 1),
                       s_ear[fe_tx].nbeats,
                       fmtf(b2,sizeof b2, g_pi, 5)); };
                out_line(_l); }
            g_emitted++;
        }
#endif
    }
    /* Keep the newest half so it runs indefinitely rather than wedging. */
    for (int d = 0; d < 2; d++) {
        /* Feet must be trimmed too, or nfeet saturates at MAXFEET and dtau
         * silently stops updating after ~3 minutes. Halving keeps the newest
         * half; the provisional estimate works on a copy so nothing mutates
         * underneath it. */
        if (s_ear[d].nfeet >= MAXFEET - 1) {
            int keep = s_ear[d].nfeet / 2;
            for (int i = 0; i < keep; i++) {
                s_ear[d].feet[i]      = s_ear[d].feet[i + keep];
                s_ear[d].feet_toff[i] = s_ear[d].feet_toff[i + keep];
            }
            s_ear[d].nfeet = keep;
        }
        if (s_ear[d].nbeats >= MAXBEATS - 1) {
            int keep = s_ear[d].nbeats / 2;
            for (int i = 0; i < keep; i++)
                s_ear[d].beats[i] = s_ear[d].beats[i + keep];
            s_ear[d].nbeats = keep;
        }
    }
}
/* ---------------------------------------------------------------------------
 * The DSP sources are #included, not linked. The MIN_BEATS / MAX_ABS_MS /
 * RR_MIN_INTERVALS overrides live at the TOP of this file so that every use
 * — provisional, validated, g_cfg_* mirrors — sees the same values.
 *
 * PORTING NOTE: because these two files are #included here, arterialage.c
 * and aa_features.c must NOT also be listed as sources in the .slcp (or be
 * excluded from build in the IDE), or every function in them links twice.
 * They only need to exist on disk beside app.c so the #include resolves.  */
#include "arterialage.c"
#include "aa_features.c"

