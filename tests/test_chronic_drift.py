"""Tests for chronic (per-segment) drift correction.

See CHRONIC_DRIFT_MODE.md for the design rationale.

"""

import numpy as np
import pytest
import torch
from scipy.sparse import coo_matrix

from kilosort import datashift, io


np.random.seed(9876)


def _reference_bin_spikes(ops, st):
    """The pre-change implementation of `bin_spikes`, kept as a reference.

    `datashift.bin_spikes` must stay bit-identical to this when no `group_id`
    is given.

    """
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth']
    dmin = ymin-1
    dmax = 1 + np.ceil((ymax-dmin)/dd).astype('int32')
    Nbatches = ops['Nbatches']
    batch_id = st[:,4].copy()

    F = np.zeros((Nbatches, dmax, 20))
    for t in range(ops['Nbatches']):
        ix = (batch_id==t).nonzero()[0]
        sst = st[ix]
        dep = sst[:,1] - dmin
        amp = np.log10(np.minimum(99, sst[:,2])) - np.log10(ops['Th_universal'])
        amp = amp / (np.log10(100)-np.log10(ops['Th_universal']))
        rows = (dep/dd).astype('int32')
        cols = (1e-5 + amp * 20).astype('int32')
        cou = np.ones(len(ix))
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))
        F[t] = np.log2(1+M.todense())

    ysamp = dmin + dd * np.arange(dmax) - dd/2

    return F, ysamp


def _fake_ops(n_batches=8, nblocks=2):
    """Minimal `ops` for the binning and registration functions."""
    yc = np.repeat(np.arange(0, 200, 20).astype('float64'), 2)
    xc = np.tile([0., 16.], yc.size//2)
    return {
        'yc': yc, 'xc': xc, 'binning_depth': 5, 'Th_universal': 9,
        'Nbatches': n_batches, 'nblocks': nblocks,
        'drift_smoothing': [0.5, 0.5, 0.5], 'sig_interp': 20,
        }


def _fake_spikes(ops, spikes_per_batch=200, depth_offset=None):
    """Synthetic `st` with the columns `bin_spikes` uses.

    `depth_offset` is added to the spike depths of each batch, so that a known
    drift can be injected.

    """
    nb = ops['Nbatches']
    if depth_offset is None:
        depth_offset = np.zeros(nb)
    ymin, ymax = ops['yc'].min(), ops['yc'].max()

    st = np.zeros((nb*spikes_per_batch, 6))
    for b in range(nb):
        sl = slice(b*spikes_per_batch, (b+1)*spikes_per_batch)
        # A few depth "hot spots" make a fingerprint with real structure,
        # which is what the registration keys on.
        centers = np.array([30., 90., 150.])
        y = np.repeat(centers, spikes_per_batch//3 + 1)[:spikes_per_batch]
        y = y + np.random.randn(spikes_per_batch)*3 + depth_offset[b]
        st[sl,0] = b                                     # time (unused here)
        st[sl,1] = np.clip(y, ymin, ymax)                # depth
        st[sl,2] = 10 + np.random.rand(spikes_per_batch)*50   # amplitude
        st[sl,4] = b                                     # batch index

    return st


class StubFile:
    """Stand-in for `io.BinaryFiltered` with just what `segment_batches` uses."""
    def __init__(self, NT=100, imin=0, n_batches=10, batch_downsampling=1):
        self.NT = NT
        self.imin = imin
        self.n_batches = n_batches
        self.batch_downsampling = batch_downsampling


class TestBinSpikes:
    def test_matches_old_implementation(self):
        # With `group_id=None`, results must be bit-identical to the old code.
        ops = _fake_ops()
        st = _fake_spikes(ops)
        F_ref, ysamp_ref = _reference_bin_spikes(ops, st)
        F_new, ysamp_new = datashift.bin_spikes(ops, st)

        assert np.array_equal(F_ref, F_new)
        assert np.array_equal(ysamp_ref, ysamp_new)

    def test_empty_batches(self):
        # Batches with no spikes must still produce an all-zero fingerprint.
        ops = _fake_ops(n_batches=6)
        st = _fake_spikes(ops)
        st = st[st[:,4] != 3]
        F_ref, _ = _reference_bin_spikes(ops, st)
        F_new, _ = datashift.bin_spikes(ops, st)

        assert np.array_equal(F_ref, F_new)
        assert np.all(F_new[3] == 0)

    def test_grouping_pools_counts(self):
        # Pooling batches {0,1} and {2,3} must equal binning their spikes
        # together, i.e. log2(1 + sum of the per-batch counts).
        ops = _fake_ops(n_batches=4)
        st = _fake_spikes(ops)
        F_batch, _ = datashift.bin_spikes(ops, st)
        gid = (st[:,4] // 2).astype('int64')
        F_pool, _ = datashift.bin_spikes(ops, st, group_id=gid, n_groups=2)

        for s in range(2):
            counts = (2**F_batch[2*s] - 1) + (2**F_batch[2*s + 1] - 1)
            assert np.allclose(F_pool[s], np.log2(1 + counts))

    def test_unsorted_group_id_raises(self):
        ops = _fake_ops(n_batches=4)
        st = _fake_spikes(ops)
        gid = np.zeros(st.shape[0], dtype='int64')
        gid[0] = 1  # not non-decreasing
        with pytest.raises(ValueError):
            datashift.bin_spikes(ops, st, group_id=gid, n_groups=2)


class TestBlockIndices:
    def test_shape_and_overlap(self):
        ifirst, ilast = datashift.block_indices(100, 5)
        assert ifirst.size == 2*5 - 1
        assert np.all(ilast - ifirst == 100//5)
        assert ifirst[0] == 0
        assert ilast[-1] == 100


class TestAlignBlock2:
    def test_shape_from_F_not_ops(self):
        # `align_block2` must take its length from `F`, not `ops['Nbatches']`,
        # so that it can be reused on per-segment fingerprints.
        ops = _fake_ops(n_batches=8, nblocks=2)
        st = _fake_spikes(ops)
        F, ysamp = datashift.bin_spikes(ops, st)
        F = F[:3]  # fewer fingerprints than ops['Nbatches']
        imin, yblk, _, _ = datashift.align_block2(
            F, ysamp, ops, device=torch.device('cpu')
            )

        assert imin.shape == (3, 2*ops['nblocks'] - 1)
        assert yblk.shape == (2*ops['nblocks'] - 1,)

    def test_too_few_fingerprints_raises(self):
        ops = _fake_ops(n_batches=8)
        st = _fake_spikes(ops)
        F, ysamp = datashift.bin_spikes(ops, st)
        with pytest.raises(ValueError):
            datashift.align_block2(F[:1], ysamp, ops,
                                   device=torch.device('cpu'))

    def test_recovers_known_step(self):
        # Half the batches are shifted down by a known amount. The estimated
        # shift must reproduce that step (up to the sign convention, which is
        # the same for every block).
        ops = _fake_ops(n_batches=12, nblocks=1)
        step = 20.0
        offsets = np.zeros(12)
        offsets[6:] = step
        st = _fake_spikes(ops, spikes_per_batch=400, depth_offset=offsets)
        F, ysamp = datashift.bin_spikes(ops, st)
        imin, _, _, _ = datashift.align_block2(
            F, ysamp, ops, device=torch.device('cpu'), drift_smoothing=[0.5, 0, 0.5]
            )
        dshift = imin*ops['binning_depth']

        estimated = dshift[6:].mean() - dshift[:6].mean()
        assert np.abs(np.abs(estimated) - step) <= ops['binning_depth']


class TestSegmentBatches:
    def test_boundary_on_batch_edge(self):
        # A boundary that lands exactly on a batch edge straddles nothing.
        ops = _fake_ops()
        ops['drift_segment_starts'] = np.array([0, 500], dtype='int64')
        bfile = StubFile(NT=100, imin=0, n_batches=10)
        seg, straddles, n = datashift.segment_batches(ops, bfile)

        assert n == 2
        assert np.array_equal(seg, np.array([0]*5 + [1]*5))
        assert not straddles.any()

    def test_boundary_mid_batch(self):
        # A boundary inside a batch flags it, and it stays with the earlier
        # segment (the one containing its first sample).
        ops = _fake_ops()
        ops['drift_segment_starts'] = np.array([0, 550], dtype='int64')
        bfile = StubFile(NT=100, imin=0, n_batches=10)
        seg, straddles, n = datashift.segment_batches(ops, bfile)

        assert n == 2
        assert np.array_equal(seg, np.array([0]*6 + [1]*4))
        assert np.array_equal(np.flatnonzero(straddles), [5])

    def test_tmin_offsets_batches(self):
        # Segment starts are raw sample indices, so `imin` (from tmin) must be
        # accounted for.
        ops = _fake_ops()
        ops['drift_segment_starts'] = np.array([0, 500], dtype='int64')
        bfile = StubFile(NT=100, imin=200, n_batches=8)
        seg, straddles, n = datashift.segment_batches(ops, bfile)

        # batch b covers [200 + 100b, 300 + 100b), so batch 3 starts at 500
        assert np.array_equal(seg, np.array([0]*3 + [1]*5))
        assert not straddles.any()

    def test_batch_downsampling(self):
        # `padded_batch_to_torch` multiplies the batch index by the
        # downsampling factor, so each sorted batch covers NT*ds samples.
        ops = _fake_ops()
        ops['drift_segment_starts'] = np.array([0, 400], dtype='int64')
        bfile = StubFile(NT=100, imin=0, n_batches=5, batch_downsampling=2)
        seg, straddles, n = datashift.segment_batches(ops, bfile)

        # batch b covers [200b, 200b + 200)
        assert np.array_equal(seg, np.array([0, 0, 1, 1, 1]))
        assert not straddles.any()

    def test_cropped_segment_is_relabeled(self):
        # A segment cropped away by tmax must not leave a gap in the labels.
        ops = _fake_ops()
        ops['drift_segment_starts'] = np.array([0, 300, 5000], dtype='int64')
        bfile = StubFile(NT=100, imin=0, n_batches=8)
        seg, straddles, n = datashift.segment_batches(ops, bfile)

        assert n == 2
        assert np.array_equal(np.unique(seg), [0, 1])
        assert np.array_equal(ops['drift_segment_used'], [0, 1])
        assert np.array_equal(ops['drift_segment_starts_used'], [0, 300])

    def test_fewer_than_two_segments_raises(self):
        ops = _fake_ops()
        ops['drift_segment_starts'] = np.array([0, 5000], dtype='int64')
        bfile = StubFile(NT=100, imin=0, n_batches=8)
        with pytest.raises(ValueError):
            datashift.segment_batches(ops, bfile)


class TestLoadDriftSegments:
    def test_newline_separated(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0\n1000\n2000\n')
        assert np.array_equal(io.load_drift_segments(p), [0, 1000, 2000])

    def test_comma_and_space_separated(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0, 1000 2000,3000')
        assert np.array_equal(io.load_drift_segments(str(p)),
                              [0, 1000, 2000, 3000])

    def test_list_input(self):
        assert np.array_equal(io.load_drift_segments([0, 10, 20]), [0, 10, 20])

    def test_prepends_zero(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('1000 2000')
        starts = io.load_drift_segments(p)
        assert np.array_equal(starts, [0, 1000, 2000])

    def test_non_monotonic_raises(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 2000 1000')
        with pytest.raises(ValueError):
            io.load_drift_segments(p)

    def test_duplicate_raises(self):
        with pytest.raises(ValueError):
            io.load_drift_segments([0, 1000, 1000])

    def test_non_integer_raises(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 1000.5')
        with pytest.raises(ValueError):
            io.load_drift_segments(p)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            io.load_drift_segments([-100, 0, 100])

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('   \n')
        with pytest.raises(ValueError):
            io.load_drift_segments(p)

    def test_unparseable_raises(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 abc')
        with pytest.raises(ValueError):
            io.load_drift_segments(p)

    def test_dtype_is_int64(self, tmp_path):
        p = tmp_path / 'segments.txt'
        p.write_text('0 108000000 215700000')
        assert io.load_drift_segments(p).dtype == np.int64


class TestSegmentResidualDrift:
    def test_zero_residual_for_constant_shift(self):
        # Drift is genuinely constant within each segment, so every batch
        # should register to its segment's fingerprint at ~0 offset.
        ops = _fake_ops(n_batches=12, nblocks=1)
        offsets = np.zeros(12)
        offsets[6:] = 20.0
        st = _fake_spikes(ops, spikes_per_batch=400, depth_offset=offsets)
        seg = np.array([0]*6 + [1]*6, dtype='int64')
        straddles = np.zeros(12, dtype=bool)
        F, ysamp = datashift.bin_spikes(ops, st, group_id=seg[st[:,4].astype(int)],
                                        n_groups=2)
        ops = datashift.segment_residual_drift(
            ops, st, seg, straddles, F, ysamp, device=torch.device('cpu')
            )

        assert ops['drift_residual'].shape == (12, 2*ops['nblocks'] - 1)
        assert np.array_equal(ops['drift_residual_batches'], np.arange(12))
        assert ops['drift_residual_summary'].shape == (2, 1, 4)
        # median residual should be within one binning bin of zero
        assert np.all(np.abs(ops['drift_residual_summary'][:,:,0])
                      <= ops['binning_depth'])

    def test_straddling_batches_are_skipped(self):
        ops = _fake_ops(n_batches=12, nblocks=1)
        st = _fake_spikes(ops, spikes_per_batch=200)
        seg = np.array([0]*6 + [1]*6, dtype='int64')
        straddles = np.zeros(12, dtype=bool)
        straddles[[5, 11]] = True
        F, ysamp = datashift.bin_spikes(ops, st, group_id=seg[st[:,4].astype(int)],
                                        n_groups=2)
        ops = datashift.segment_residual_drift(
            ops, st, seg, straddles, F, ysamp, device=torch.device('cpu')
            )

        assert np.array_equal(ops['drift_residual_batches'],
                              [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])

    def test_subsampling_is_bounded(self):
        ops = _fake_ops(n_batches=12, nblocks=1)
        st = _fake_spikes(ops, spikes_per_batch=100)
        seg = np.array([0]*6 + [1]*6, dtype='int64')
        straddles = np.zeros(12, dtype=bool)
        F, ysamp = datashift.bin_spikes(ops, st, group_id=seg[st[:,4].astype(int)],
                                        n_groups=2)
        ops = datashift.segment_residual_drift(
            ops, st, seg, straddles, F, ysamp, device=torch.device('cpu'),
            max_batches=3
            )

        assert ops['drift_residual'].shape[0] <= 6


class TestPlots:
    def test_segment_boundary_times(self):
        from kilosort import plots

        ops = {
            'batch_to_segment': np.array([0]*5 + [1]*5),
            'settings': {'fs': 30000, 'batch_size': 60000},
            }
        t = plots.segment_boundary_times(ops, tmin=0)
        assert np.allclose(t, [10.0])

        t = plots.segment_boundary_times(ops, tmin=4.0)
        assert np.allclose(t, [14.0])

    def test_no_segments_returns_none(self):
        from kilosort import plots
        assert plots.segment_boundary_times({'batch_to_segment': None}) is None
        assert plots.segment_boundary_times({}) is None


@pytest.mark.slow
class TestEndToEnd:
    """Runs on the small test binary. Use `pytest --runslow` to include these."""

    def test_no_regression_on_real_spikes(self, saved_ops, bfile, torch_device):
        # With no segments, the refactored `bin_spikes` / `align_block2` must
        # reproduce the previous drift estimate exactly. Comparing the two
        # implementations on the same detected spikes keeps this independent of
        # the machine the reference results were generated on.
        from kilosort import spikedetect, DEFAULT_SETTINGS

        ops = {**DEFAULT_SETTINGS, **saved_ops}
        st, _, ops = spikedetect.run(ops, bfile, device=torch_device)

        F_new, ysamp_new = datashift.bin_spikes(ops, st)
        F_ref, ysamp_ref = _reference_bin_spikes(ops, st)
        assert np.array_equal(F_new, F_ref)
        assert np.array_equal(ysamp_new, ysamp_ref)

        imin_new, yblk_new, _, _ = datashift.align_block2(
            F_new, ysamp_new, ops, device=torch_device
            )
        imin_ref, yblk_ref, _, _ = datashift.align_block2(
            F_ref, ysamp_ref, ops, device=torch_device
            )
        assert np.array_equal(imin_new, imin_ref)
        assert np.array_equal(yblk_new, yblk_ref)
        assert imin_new.shape == (ops['Nbatches'], 2*ops['nblocks'] - 1)

    def test_chronic_run(self, data_directory, torch_device, tmp_path):
        # Full pipeline with the recording split into 3 artificial segments.
        from kilosort import run_kilosort

        bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
        n_samples = io.get_total_samples(bin_file, 385, 'int16')
        starts = [0, int(n_samples/3), int(2*n_samples/3)]
        results_dir = tmp_path / 'chronic'

        ops, *_ = run_kilosort.run_kilosort(
            filename=bin_file, device=torch_device,
            settings={'n_chan_bin': 385, 'nblocks': 1,
                      'drift_segment_starts': starts},
            probe_name='NeuroPix1_default.mat', results_dir=results_dir
            )

        seg = ops['batch_to_segment']
        assert seg is not None
        assert np.unique(seg).size == 3
        assert ops['dshift'].shape == (ops['Nbatches'], 2*ops['nblocks'] - 1)
        # dshift must be *exactly* constant within each segment
        for s in np.unique(seg):
            rows = ops['dshift'][seg == s]
            assert np.all(rows == rows[0])
        assert ops['drift_residual'].shape[1] == 2*ops['nblocks'] - 1
        assert (results_dir / 'drift_segments.png').is_file()

    def test_standard_run_sets_no_segments(self, data_directory, torch_device,
                                           tmp_path):
        from kilosort import run_kilosort

        bin_file = data_directory / 'ZFM-02370_mini.imec0.ap.short.bin'
        ops, *_ = run_kilosort.run_kilosort(
            filename=bin_file, device=torch_device,
            settings={'n_chan_bin': 385, 'nblocks': 1},
            probe_name='NeuroPix1_default.mat',
            results_dir=tmp_path / 'standard'
            )

        assert ops['batch_to_segment'] is None
        assert ops['drift_segment_starts'] is None
        assert 'drift_residual' not in ops
