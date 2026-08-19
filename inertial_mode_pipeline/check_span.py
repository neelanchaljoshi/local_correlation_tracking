"""
check_span.py
-------------
Stand-alone diagnostic for the "eigenfunction is always zero" issue.

Loads your real t_array via inertial_mode_pipeline.io.load_flow_data,
applies the exact same --span_lower/--span_upper mask run_pipeline.py
uses, and reports whether the resulting frequency grid gives the
bandpass filter enough surviving bins for a non-zero result.

The frequency grid has bins spaced
Δf = 1e9 / (n_span * dt_seconds) apart. A passband of width 2*df needs
to be wide enough to contain at least a few of those bins.

bandpass_filter's Tukey window defaults to tukey_alpha=0.0 (rectangular,
no edge zeroing) on the pipeline's default code path, so a single
surviving bin already carries full weight — band.sum() >= 1 is
technically enough for a non-zero result. This script checks against a
slightly higher min_bins (default 2) purely as a statistical margin: a
lone surviving bin can land anywhere in the band and is a fragile,
noisy estimate of the mode, not because of any taper effect. (If you
explicitly pass tukey_alpha > 0 elsewhere, note that its taper zeroes
the first/last surviving bin, so a non-empty band can still collapse to
all zeros — see tests/test_span_resolution.py:TestTukeyWindowBehavior.)

Usage
-----
    python check_span.py <cent_freq> <data> [options]

Example (matching the README's default run):
    python check_span.py -171.0 hmi.m_720s_dt_1h --span_lower 2010 --span_upper 2025 --df 10

To check a shorter, riskier span:
    python check_span.py -171.0 hmi.m_720s_dt_1h --span_lower 2020 --span_upper 2021 --df 10
"""

import argparse

import numpy as np

from inertial_mode_pipeline.config import DT_SEC
from inertial_mode_pipeline.io import load_flow_data

# Bare minimum for a non-zero result under the default rectangular
# window is 1. Default here is 2 for a small margin against a single,
# fragile surviving bin.
MIN_BINS = 2


def check(t_array, span_lower, span_upper, cent_freq, df,
          dt_seconds=DT_SEC, min_bins=MIN_BINS):
    span = (t_array >= span_lower) & (t_array < span_upper)
    n_span = int(span.sum())

    if n_span == 0:
        return {
            'n_span': 0,
            'freq_resolution_nHz': float('inf'),
            'n_bins_in_band': 0,
            'ok': False,
            'reason': 'no timesteps fall inside [span_lower, span_upper) at all',
        }

    freq_nHz = np.fft.fftshift(-np.fft.fftfreq(n_span, dt_seconds) * 1e9)
    band = (freq_nHz > cent_freq - df) & (freq_nHz < cent_freq + df)
    n_bins = int(band.sum())
    freq_res = 1e9 / (n_span * dt_seconds)

    reason = None
    if n_bins == 0:
        reason = 'the passband is completely empty — no frequency bin lands inside it'
    elif n_bins < min_bins:
        reason = (
            f"the passband only has {n_bins} bin(s), fewer than the "
            f"requested margin of {min_bins}; the mode estimate would rest "
            f"on a single fragile bin — this is the resolution-starvation "
            f"mechanism behind the reported zero-eigenfunction bug")

    return {
        'n_span': n_span,
        'freq_resolution_nHz': freq_res,
        'n_bins_in_band': n_bins,
        'ok': n_bins >= min_bins,
        'reason': reason,
    }


def suggest_fix(result, min_bins=MIN_BINS):
    freq_res = result['freq_resolution_nHz']
    if freq_res == float('inf'):
        print("  -> widen --span_lower/--span_upper so some data actually falls inside it.")
        return
    # Worst-case-alignment guarantee of >= min_bins bins needs a passband
    # width (2*df) of at least min_bins * freq_res.
    df_needed = min_bins * freq_res / 2
    per_bin_res_needed = 2 * df_needed / min_bins
    print(f"  -> raise --df to at least ~{df_needed:.1f} nHz for this span length, OR")
    print(f"  -> widen the span by roughly {min_bins}x more timesteps "
          f"(down to ~{per_bin_res_needed:.2f} nHz frequency resolution) "
          f"to hit {min_bins} bins at the current --df.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cent_freq', type=float)
    p.add_argument('data', type=str)
    p.add_argument('--span_lower', type=float, default=2010)
    p.add_argument('--span_upper', type=float, default=2025)
    p.add_argument('--df', type=float, default=10.0)
    p.add_argument('--min_bins', type=int, default=MIN_BINS,
                    help='Minimum surviving passband bins required (default 2)')
    args = p.parse_args()

    data_name = args.data.replace('.', '_')
    raw = load_flow_data(data_name)
    t_array = raw['t_array']

    result = check(t_array, args.span_lower, args.span_upper, args.cent_freq,
                    args.df, min_bins=args.min_bins)

    print(f"\nspan          : [{args.span_lower}, {args.span_upper})")
    print(f"cent_freq     : {args.cent_freq} nHz   df: {args.df} nHz "
          f"(passband width {2*args.df} nHz)")
    print(f"n_span        : {result['n_span']} timesteps")
    print(f"freq resolution: {result['freq_resolution_nHz']:.3f} nHz")
    print(f"bins in band  : {result['n_bins_in_band']}  (need >= {args.min_bins})")

    if result['ok']:
        print(f"\nOK: {result['n_bins_in_band']} bins survive — should produce "
              f"a non-zero eigenfunction.")
    else:
        print(f"\nFAIL: {result['reason']}")
        if result['n_span'] > 0:
            suggest_fix(result, min_bins=args.min_bins)


if __name__ == '__main__':
    main()
