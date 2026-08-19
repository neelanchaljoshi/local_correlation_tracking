"""
tests/test_span_resolution.py
------------------------------
Regression test for the "eigenfunction is always zero" issue.

Root cause
----------
run_pipeline.py builds `span = (t_array >= span_lower) & (t_array < span_upper)`
and passes it straight into fourier.transform_to_fourier, which uses
N_span = span.sum() as the FFT length in time. That sets the frequency
resolution of the whole pipeline:

    df_res [nHz] = 1e9 / (N_span * dt_seconds)

fourier.bandpass_filter only keeps frequency bins strictly inside
(cent_freq - df, cent_freq + df). If df_res is coarser than the passband
width (2*df), zero frequency bins can survive the mask, the filtered
signal is exactly zero, and extract_eigenfunction silently returns an
all-zero eigenfunction with no error or warning anywhere in the chain.

Note on the Tukey taper (historical)
-------------------------------------
An earlier version of bandpass_filter used tukey_alpha=0.1 by default,
whose taper always forces the first and last surviving bin to exactly
zero (see TestTukeyWindowBehavior below). That made even a *non-empty*
band (band.sum() == 1 or 2) collapse to all zeros, which is what
actually caused the originally-reported 4-year/default-df failure.

As of commit a89b2a1 ("changed the alpha in tukey from 0.1 to 0.0"),
bandpass_filter's default tukey_alpha is 0.0, i.e. a rectangular window
with no edge zeroing. That mechanism no longer fires on the pipeline's
default code path (extract_m_slice -> bandpass_filter never passes
tukey_alpha explicitly). The remaining — and now sole — failure mode on
the default path is pure resolution starvation: band.sum() == 0. This
file's end-to-end tests assert against that current behavior, while
keeping unit tests that document the old edge-zeroing mechanism for
anyone who opts back into tukey_alpha > 0.

This test drives fourier.transform_to_fourier + bandpass_filter with the
exact same call pattern run_pipeline.py uses, for a range of span lengths,
and asserts that band occupancy (and therefore a non-zero filtered
signal) only survives once N_span is large enough. It also exposes
reusable helpers (passband_bin_count, min_df_for_span, build_min_df_table)
you can call directly to check a proposed --span_lower/--span_upper/--df
combination *before* launching a full run, and writes a markdown
reference table to tests/min_df_reference.md.
"""

import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.fourier import transform_to_fourier, bandpass_filter
from inertial_mode_pipeline.config import DT_SEC


def passband_bin_count(n_span: int, cent_freq: float, df: float,
                        dt_seconds: float = DT_SEC) -> dict:
    """
    Compute the frequency grid the pipeline would build for n_span
    timesteps at dt_seconds cadence, and report how many bins survive
    the (cent_freq - df, cent_freq + df) bandpass.

    Returns a dict with n_span, freq_resolution_nHz, band_width_nHz,
    n_bins_in_band, and ok (bool).
    """
    freq_nHz = np.fft.fftshift(-np.fft.fftfreq(n_span, dt_seconds) * 1e9)
    band = (freq_nHz > cent_freq - df) & (freq_nHz < cent_freq + df)
    freq_res = dt_seconds and (1e9 / (n_span * dt_seconds))
    return {
        'n_span': n_span,
        'freq_resolution_nHz': freq_res,
        'band_width_nHz': 2 * df,
        'n_bins_in_band': int(band.sum()),
        'ok': bool(band.sum() > 0),
    }


def min_df_for_span(span_years: float, dt_seconds: float = DT_SEC,
                     records_per_day: float = 24 * 3600 / DT_SEC,
                     min_bins: int = 2, margin: float = 1.2) -> dict:
    """
    Guaranteed-safe minimum --df for a given span length, independent of
    grid alignment.

    Grid spacing. The frequency grid has bins spaced
    Δf = 1e9 / (n_span * dt_seconds) apart. By the pigeonhole principle,
    an interval of width W is guaranteed to contain at least k bins, for
    any alignment, once W >= k * Δf.

    On the pipeline's default code path (bandpass_filter's tukey_alpha
    defaults to 0.0, i.e. rectangular — see TestTukeyWindowBehavior), a
    single surviving bin already carries full weight and is technically
    enough for a non-zero result: min_bins=1 would be the bare
    mathematical minimum. The default here is min_bins=2, purely as a
    statistical margin — a lone surviving bin can land anywhere in the
    band and is a fragile, noisy estimate of the mode, not because of
    any taper effect.

    Since the passband width is 2*df, the guarantee becomes:

        2 * df >= min_bins * Δf   i.e.   df >= min_bins * Δf / 2

    `margin` inflates that slightly further (default +20%) to stay
    clear of the exact boundary, where bandpass_filter's strict
    '>' / '<' comparisons can exclude a bin sitting exactly at the edge.

    Returns a dict with span_years, n_span, freq_resolution_nHz,
    min_df_guaranteed_nHz (bare, min_bins-bin guarantee), and
    recommended_df_nHz (with margin).
    """
    n_span = int(round(span_years * 365.25 * records_per_day))
    freq_res = 1e9 / (n_span * dt_seconds)
    min_df = min_bins * freq_res / 2
    return {
        'span_years': span_years,
        'n_span': n_span,
        'freq_resolution_nHz': freq_res,
        'min_df_guaranteed_nHz': min_df,
        'recommended_df_nHz': min_df * margin,
    }


def build_min_df_table(span_years_list=None, dt_seconds: float = DT_SEC,
                        margin: float = 1.2) -> list:
    """Build the full reference table as a list of dicts, one per span length."""
    if span_years_list is None:
        span_years_list = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15]
    return [min_df_for_span(y, dt_seconds=dt_seconds, margin=margin)
            for y in span_years_list]


def format_min_df_table(rows: list) -> str:
    """Render build_min_df_table's output as a Markdown table."""
    header = ('| Span (years) | N_span (approx.) | Δf resolution (nHz) | '
               'Min. --df (guaranteed) | Recommended --df |')
    sep = '|---|---|---|---|---|'
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['span_years']:g} | {r['n_span']:,} | "
            f"{r['freq_resolution_nHz']:.2f} | "
            f"{r['min_df_guaranteed_nHz']:.2f} | "
            f"{r['recommended_df_nHz']:.1f} |")
    return '\n'.join(lines)


def make_synthetic_flow(n_span: int, nlat: int = 10, nlng_stony: int = 37,
                         cent_freq: float = -171.0, m: int = 2,
                         dt_seconds: float = DT_SEC, seed: int = 0):
    """
    Build a synthetic (nt, nlat, nlng) u_phi-like array containing a pure
    tone at `cent_freq` nHz in azimuthal order `m`, so a working pipeline
    should recover a non-zero eigenfunction, and a starved one should not.

    cft/cfl are shaped (nt, nlat, 1) to match geometry.get_correction_factor's
    real output — see transform_to_fourier, which indexes cfl/cft with the
    same `span` boolean mask used for the time axis.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_span) * dt_seconds  # seconds
    omega = 2 * np.pi * cent_freq * 1e-9  # rad/s
    lon_phase = m * np.linspace(-np.pi/2, np.pi/2, nlng_stony)

    tone = np.cos(omega * t)[:, None, None] * np.cos(lon_phase)[None, None, :]
    noise = 0.01 * rng.standard_normal((n_span, nlat, nlng_stony))
    arr = tone * np.ones((1, nlat, 1)) + noise

    # crln=0: the synthetic tone is injected already "in" the Carrington
    # frame, so transform_to_fourier's per-timestep de-rotation phase
    # (exp(-1j * m * crln)) is a no-op and doesn't shift our injected
    # frequency. Real crln progresses at the solar rotation rate and is
    # exactly what that de-rotation step is meant to undo.
    crln = np.zeros(n_span)
    cft = np.ones((n_span, nlat, 1))
    cfl = np.ones((n_span, nlat, 1))
    return arr, crln, cft, cfl


class TestMinDfReferenceTable(unittest.TestCase):
    """
    Generates a span-length -> minimum-safe-df reference table, writes it
    to tests/min_df_reference.md, and sanity-checks the underlying
    relationship. Run this file (or the whole suite) once and the table
    file is there for anyone using the pipeline to consult directly —
    no need to run check_span.py or do the arithmetic by hand.
    """

    def test_table_is_monotonic_and_matches_current_default_behavior(self):
        rows = build_min_df_table()
        # More span -> always a smaller (or equal) minimum required df.
        for prev, cur in zip(rows, rows[1:]):
            self.assertLessEqual(cur['min_df_guaranteed_nHz'],
                                  prev['min_df_guaranteed_nHz'])

        # With the current rectangular default window (tukey_alpha=0.0),
        # a 1-year span is the still-failing case at the default --df 10
        # (0 bins survive, see TestSpanResolutionEndToEnd) — it should
        # need noticeably more than 10 nHz to be guaranteed >= min_bins.
        one_year = next(r for r in rows if r['span_years'] == 1)
        self.assertGreater(one_year['min_df_guaranteed_nHz'], 10.0)

        # 15 years (the README default span) should need well under 10.
        fifteen_year = next(r for r in rows if r['span_years'] == 15)
        self.assertLess(fifteen_year['min_df_guaranteed_nHz'], 10.0)

    def test_write_reference_table_file(self):
        rows = build_min_df_table()
        table_md = format_min_df_table(rows)
        out_path = pathlib.Path(__file__).parent / 'min_df_reference.md'
        out_path.write_text(
            '# Minimum --df by span length\n\n'
            'Generated by tests/test_span_resolution.py — '
            'run `pytest tests/test_span_resolution.py` to refresh.\n\n'
            'Guaranteed minimum: the largest df below which an empty '
            'passband (and therefore a zero eigenfunction) is possible '
            'purely from grid alignment, for ANY cent_freq. Assumes the '
            'pipeline default of tukey_alpha=0.0 (rectangular window); '
            'if a caller opts into tukey_alpha > 0 the taper zeroes the '
            'first/last surviving bin and a larger margin is needed — '
            'see TestTukeyWindowBehavior in test_span_resolution.py.\n'
            'Recommended: guaranteed minimum plus a 20% safety margin.\n\n'
            + table_md + '\n')
        self.assertTrue(out_path.exists())


class TestTukeyWindowBehavior(unittest.TestCase):
    """
    Pins down bandpass_filter's Tukey window behavior at both the
    current default (tukey_alpha=0.0, rectangular — no edge zeroing) and
    the historical default (tukey_alpha=0.1, which always forced the
    first and last surviving bin to exactly zero). The rectangular
    default is what the pipeline actually uses on its default code path
    (extract_m_slice -> bandpass_filter never passes tukey_alpha), so
    band.sum() > 0 is now sufficient for a non-zero filtered signal.
    """

    def test_default_alpha_is_rectangular_no_edge_zeroing(self):
        from inertial_mode_pipeline.fourier import tukeywin
        for n in (1, 2, 3):
            w = tukeywin(n)
            np.testing.assert_array_equal(
                w, np.ones(n),
                err_msg=f"tukeywin({n}) with the default alpha should be "
                        f"all-ones (rectangular), got {w}")

    def test_explicit_alpha_still_tapers_edges_to_zero(self):
        """
        Historical/opt-in behavior: a caller who explicitly passes
        tukey_alpha > 0 still gets the old edge-zeroing taper. This is
        what caused the originally-reported bug back when 0.1 was the
        default (see TestSpanResolutionEndToEnd
        .test_would_have_failed_under_old_tukey_alpha_default).
        """
        from inertial_mode_pipeline.fourier import tukeywin
        for n in (1, 2):
            w = tukeywin(n, 0.1)
            self.assertTrue(np.all(w == 0),
                f"tukeywin({n}, 0.1) should be entirely zero, got {w}")
        w3 = tukeywin(3, 0.1)
        self.assertEqual(np.count_nonzero(w3), 1)
        self.assertAlmostEqual(w3[1], 1.0)

    def test_bandpass_filter_default_keeps_a_two_bin_window_nonzero(self):
        """Direct check on bandpass_filter itself (not just tukeywin)."""
        freq_nHz = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        sig = np.ones((5, 1), dtype=complex)
        filt, _ = bandpass_filter(sig, sig, freq_nHz, cent_freq=0.0, df=7.0)
        # band is (-7, 7) -> bins at -5 and 5 survive (2 bins)
        self.assertTrue(np.all(np.abs(filt[freq_nHz == -5.0]) > 0))
        self.assertTrue(np.all(np.abs(filt[freq_nHz == 5.0]) > 0))

    def test_four_year_span_has_exactly_two_band_bins(self):
        """
        Frequency-grid arithmetic behind the originally-reported 4-year
        failure. Whether this actually zeroes the output now depends on
        tukey_alpha (see TestSpanResolutionEndToEnd for both outcomes).
        """
        n_span = int(4 * 365.25 * 4)
        result = passband_bin_count(n_span, cent_freq=-171.0, df=10.0)
        self.assertEqual(result['n_bins_in_band'], 2,
            "4-year span at default df=10 should have exactly 2 band bins")

    def test_five_year_span_has_exactly_three_band_bins(self):
        n_span = int(5 * 365.25 * 4)
        result = passband_bin_count(n_span, cent_freq=-171.0, df=10.0)
        self.assertEqual(result['n_bins_in_band'], 3,
            "5-year span at default df=10 should have exactly 3 band bins")


class TestSpanResolutionEmptyPassband(unittest.TestCase):
    """
    Direct reproduction of the pure resolution-starvation mechanism
    using the helper above — no real HMI/LCT data required, and no
    dependency on tukey_alpha (an empty band is empty regardless of the
    window applied to it).
    """

    def setUp(self):
        self.cent_freq = -171.0
        self.df = 10.0  # matches run_pipeline.py's --df default

    def test_wide_span_has_occupied_band(self):
        # ~15 years at 6h cadence, matching the README default span
        n_span = int(15 * 365.25 * 4)
        result = passband_bin_count(n_span, self.cent_freq, self.df)
        self.assertTrue(result['ok'],
            f"Expected a non-empty passband for a wide span, got {result}")
        self.assertGreater(result['n_bins_in_band'], 1)

    def test_narrow_span_can_starve_the_band(self):
        # ~3 months at 6h cadence — deliberately short
        n_span = int(0.25 * 365.25 * 4)
        result = passband_bin_count(n_span, self.cent_freq, self.df)
        self.assertFalse(result['ok'],
            f"Expected an empty passband for a narrow span, got {result}")
        self.assertEqual(result['n_bins_in_band'], 0)

    def test_crossover_point_matches_theory(self):
        """
        The band is guaranteed empty once the frequency resolution is
        coarser than the full passband width (2*df) *and* no grid point
        happens to land inside by chance alignment; it is guaranteed
        non-empty once resolution is comfortably finer than the band.
        We only assert the unambiguous ends of the sweep — points near
        the crossover can go either way depending on grid alignment
        with cent_freq, which is itself part of why this bug is so easy
        to hit unpredictably in practice.
        """
        band_width = 2 * self.df
        for n_span in [50, 200]:
            result = passband_bin_count(n_span, self.cent_freq, self.df)
            self.assertGreater(result['freq_resolution_nHz'], 5 * band_width)
            self.assertFalse(result['ok'],
                f"n_span={n_span}: resolution {result['freq_resolution_nHz']:.2f} nHz "
                f"is far coarser than the {band_width} nHz band, expected an empty passband")
        for n_span in [12800, 51200]:
            result = passband_bin_count(n_span, self.cent_freq, self.df)
            self.assertLess(result['freq_resolution_nHz'], band_width / 5)
            self.assertTrue(result['ok'],
                f"n_span={n_span}: resolution {result['freq_resolution_nHz']:.2f} nHz "
                f"is far finer than the {band_width} nHz band, expected an occupied passband")

    def test_widening_df_recovers_a_narrow_span(self):
        """A short span can be rescued by widening --df to match."""
        n_span = int(0.25 * 365.25 * 4)  # same starved span as above
        starved = passband_bin_count(n_span, self.cent_freq, self.df)
        self.assertFalse(starved['ok'])

        rescued = passband_bin_count(n_span, self.cent_freq, df=100.0)
        self.assertTrue(rescued['ok'],
            "Widening df should recover at least one bin for the same short span")


class TestSpanResolutionEndToEnd(unittest.TestCase):
    """
    Runs an injected pure tone through the real transform_to_fourier +
    bandpass_filter functions to show the same failure at the array
    level (not just the frequency-label arithmetic), against the
    pipeline's actual default code path (tukey_alpha=0.0).
    """

    def _bandpass_energy(self, n_span, cent_freq=-171.0, df=10.0, m=2,
                          tukey_alpha=None):
        arr, crln, cft, cfl = make_synthetic_flow(
            n_span, cent_freq=cent_freq, m=m)
        span_mask = np.ones(n_span, dtype=bool)
        ft, freq_nHz = transform_to_fourier(arr, crln, cft, cfl, span_mask)
        kwargs = {} if tukey_alpha is None else {'tukey_alpha': tukey_alpha}
        uphi_filt, _ = bandpass_filter(
            ft[:, :, m], ft[:, :, m], freq_nHz, cent_freq, df=df, **kwargs)
        return np.sum(np.abs(uphi_filt) ** 2)

    def test_one_year_span_is_still_starved_at_default_df(self):
        """
        With the current default (tukey_alpha=0.0), pure resolution
        starvation (0 band bins) is the only remaining way to get an
        all-zero eigenfunction. A 1-year span at 6h cadence with the
        default --df 10 lands in exactly that regime.
        """
        n_span = int(1 * 365.25 * 4)
        energy = self._bandpass_energy(n_span)
        self.assertEqual(energy, 0.0,
            "A 1-year span at the default --df 10 is expected to fully "
            "starve the default ±10 nHz passband (0 surviving bins)")

    def test_four_year_span_now_succeeds_with_default_alpha(self):
        """
        The specific case originally reported as failing (4-year span,
        default --df 10). Since commit a89b2a1 changed bandpass_filter's
        default tukey_alpha to 0.0, this span's 2 surviving bins are no
        longer zeroed by the taper, so it now correctly recovers signal.
        """
        n_span = int(4 * 365.25 * 4)
        energy = self._bandpass_energy(n_span)
        self.assertGreater(energy, 0.0,
            "A 4-year span should retain signal energy in the passband "
            "under the current rectangular-window default")

    def test_would_have_failed_under_old_tukey_alpha_default(self):
        """
        Regression/documentation test: reproduces the originally-reported
        bug exactly as it behaved under the old tukey_alpha=0.1 default,
        by passing it explicitly. Protects against silently reintroducing
        the failure if bandpass_filter's default is ever changed back.
        """
        n_span = int(4 * 365.25 * 4)
        energy = self._bandpass_energy(n_span, tukey_alpha=0.1)
        self.assertEqual(energy, 0.0,
            "Under the old tukey_alpha=0.1 default, a 4-year span's 2 "
            "surviving bins should both be zeroed by the Tukey taper")

    def test_five_year_span_is_enough_even_under_old_tukey_alpha(self):
        n_span = int(5 * 365.25 * 4)
        energy = self._bandpass_energy(n_span, tukey_alpha=0.1)
        self.assertGreater(energy, 0.0,
            "A 5-year span should retain signal energy even with the "
            "old tukey_alpha=0.1 taper (3 bins, 1 survives)")

    def test_very_narrow_span_loses_all_energy(self):
        n_span = int(0.1 * 365.25 * 4)  # ~5 weeks
        energy = self._bandpass_energy(n_span)
        self.assertEqual(energy, 0.0,
            "A ~5-week span is expected to fully starve the default ±10 nHz passband")


if __name__ == '__main__':
    unittest.main()
