import logging
logger = logging.getLogger(__name__)

import warnings

from scipy.sparse import coo_matrix
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from kilosort import spikedetect


def bin_spikes(ops, st, group_id=None, n_groups=None):
    """Bin spikes into (depth x log-amplitude) fingerprints, one per group.

    By default each batch forms its own group, so that every batch gets its own
    2D histogram ("fingerprint") of spikes over depth and log-amplitude. For
    chronic drift correction, `group_id` is instead the recording segment each
    spike belongs to, which pools a whole segment into a single fingerprint.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        Spike times array with 6 columns, as returned by `spikedetect.run`.
    group_id : np.ndarray; optional.
        Integer group index for each spike, **sorted non-decreasing**. Defaults
        to the batch index, `st[:,4]`, reproducing the original behavior.
    n_groups : int; optional.
        Number of groups. Defaults to `ops['Nbatches']`.

    Returns
    -------
    F : np.ndarray
        Fingerprints, shape (n_groups, dmax, 20).
    ysamp : np.ndarray
        Center of each vertical sampling bin, shape (dmax,).

    """

    # the bin edges are based on min and max of channel y positions
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth'] # binning width in depth

    # start 1um below the lowest channel
    dmin = ymin-1

    # dmax is how many bins to use
    dmax = 1 + np.ceil((ymax-dmin)/dd).astype('int32')

    if group_id is None:
        group_id = st[:,4]
        n_groups = ops['Nbatches']
    group_id = np.asarray(group_id).astype('int64')
    if group_id.size > 0 and np.any(np.diff(group_id) < 0):
        raise ValueError('bin_spikes requires `group_id` sorted non-decreasing')

    # Group boundaries in O(n_groups * log n_spikes). The previous
    # implementation rescanned the full spike array once per group, which is
    # O(n_groups * n_spikes) and becomes the dominant cost for long recordings.
    edges = np.searchsorted(group_id, np.arange(n_groups+1), side='left')

    # always use 20 bins for amplitude binning
    F = np.zeros((n_groups, dmax, 20))
    for t in range(n_groups):
        # consider only spikes from this group
        sst = st[edges[t]:edges[t+1]]
        if sst.shape[0] == 0:
            continue

        # their depth relative to the minimum
        dep = sst[:,1] - dmin

        # the amplitude binnning is logarithmic, goes from the Th_universal minimum value to 100.
        amp = np.log10(np.minimum(99, sst[:,2])) - np.log10(ops['Th_universal'])

        # amplitudes get normalized from 0 to 1
        amp = amp / (np.log10(100)-np.log10(ops['Th_universal']))

        # rows are divided by the vertical binning depth
        rows = (dep/dd).astype('int32')

        # columns are from 0 to 20
        cols = (1e-5 + amp * 20).astype('int32')

        # for efficient binning, use sparse matrix computation in scipy
        cou = np.ones(sst.shape[0])
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))

        # the 2D histogram counts are transformed to logarithm
        F[t] = np.log2(1+M.todense())

    # center of each vertical sampling bin
    ysamp = dmin + dd * np.arange(dmax) - dd/2

    return F, ysamp


def block_indices(nybins, nblocks):
    """Get the first and last vertical bin of each registration block.

    Depth is divided into `nblocks` non-overlapping segments, plus the
    `nblocks - 1` segments which half-overlap those, for `2*nblocks - 1` blocks
    in total.

    Parameters
    ----------
    nybins : int
        Number of vertical bins in the fingerprints (`F.shape[1]`).
    nblocks : int
        Number of non-overlapping blocks, i.e. `ops['nblocks']`.

    Returns
    -------
    ifirst : np.ndarray
        First vertical bin of each block, shape (2*nblocks - 1,).
    ilast : np.ndarray
        One past the last vertical bin of each block, shape (2*nblocks - 1,).

    """

    yl = nybins//nblocks
    ifirst = np.round(np.linspace(0, nybins-yl, 2*nblocks-1)).astype('int32')
    ilast = ifirst + yl

    return ifirst, ilast


def align_block2(F, ysamp, ops, device=torch.device('cuda'),
                 drift_smoothing=None):
    """Register fingerprints to each other to estimate vertical drift.

    Parameters
    ----------
    F : np.ndarray
        Fingerprints from `bin_spikes`, shape (n, dmax, 20). Note that `n` is
        the number of registration units, which is the number of batches for
        standard drift correction and the number of segments in chronic mode.
    ysamp : np.ndarray
        Center of each vertical sampling bin.
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    device : torch.device; default=torch.device('cuda').
    drift_smoothing : list of float; optional.
        Overrides `ops['drift_smoothing']`. Used by chronic mode to disable
        smoothing along the time axis, since a segment-to-segment step is real
        signal rather than estimation noise.

    """

    if F.shape[0] < 2:
        raise ValueError('Drift registration needs at least 2 fingerprints; '
                         f'got {F.shape[0]}.')

    # number of registration units (batches, or segments in chronic mode)
    Nbatches = F.shape[0]

    # n is the maximum vertical shift allowed, in units of bins
    n = 15
    dc = np.zeros((2*n+1, Nbatches))
    dt = np.arange(-n,n+1,1)

    # batch fingerprints are mean subtracted along depth
    Fg = torch.from_numpy(F).to(device).float()
    Fg = Fg - Fg.mean(1).unsqueeze(1)

    # the template fingerprint is initialized with batch 300 if that exists
    F0 = Fg[np.minimum(300, Nbatches//2)]

    niter = 10
    dall = np.zeros((niter, Nbatches))

    # at each iteration, align each batch to the template fingerprint
    # Fg is incrementally modified, and cumulative shifts are accumulated over iterations
    for iter in range(niter):
        # for each vertical shift in the range -n to n, compute the dot product
        for t in range(len(dt)):
            Fs = torch.roll(Fg, dt[t], 1)
            dc[t] = (Fs * F0).mean(-1).mean(-1).cpu().numpy()

        # for all but the last iteration, align the batches
        if iter<niter-1:
            # the maximum dot product is the best match for each batch
            imax = np.argmax(dc, 0)

            for t in range(len(dt)):
                # for batches which have the maximum at dt[t]
                ib = imax==t

                # roll the fingerprints for those batches by dt[t]
                Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
                dall[iter, ib] = dt[t]

        # take the mean of the aligned batches. This will be the new fingerprint template.
        F0 = Fg.mean(0)


    # divide the vertical bins into nblocks non-overlapping segments, and then consider also the segments which half-overlap these segments
    nblocks = ops['nblocks']
    nybins = F.shape[1]
    ifirst, ilast = block_indices(nybins, nblocks)

    # the new nblocks is 2*nblocks - 1 due to the overlapping blocks
    nblocks = len(ifirst)
    yblk = np.zeros(nblocks,)

    # consider much smaller ranges for the fine drift correction
    n  = 5
    dt = np.arange(-n, n+1, 1)
    dcs = np.zeros((2*n+1, Nbatches, nblocks))

    # for each block in each batch, recompute the dot products with the template
    for j in range(nblocks):
        isub = np.arange(ifirst[j], ilast[j], 1)
        yblk[j] = ysamp[isub].mean()

        Fsub = Fg[:, isub]

        for t in range(len(dt)):
            Fs = torch.roll(Fsub, dt[t], 1)
            dcs[t, :, j] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()

    # upsamples the dot-product matrices by 10 to get finer estimates of vertica ldrift
    dtup = np.linspace(-n,n,2*n*10+1)

    # get 1D upsampling matrix
    Kn = kernelD(dt,dtup,1)

    # smooth the dot-product matrices across correlation, batches, and vertical offsets
    if drift_smoothing is None:
        drift_smoothing = ops['drift_smoothing']
    dcs = gaussian_filter(dcs, drift_smoothing)

    # for each block, upsample the dot-product matrix and find new max
    imin = np.zeros((Nbatches, nblocks))
    for j in range(nblocks):
        dcup = Kn.T @ dcs[:,:,j]
        imax = np.argmax(dcup, 0)

        # the new max gets added to the last iteration of dall
        dall[niter-1] = dtup[imax]

        # the cumulative shifts in dall represent the total vertical shift for each batch
        imin[:,j] = dall.sum(0)

    # Fg gets reinitialized with the un-corrected F without subtracting the mean across depth.
    Fg = torch.from_numpy(F).float()
    imax = dall[:niter-1].sum(0)

    # Fg gets aligned again to compute the non-mean subtracted fingerprint
    for t in range(len(dt)):
        ib = imax==dt[t]
        Fg[ib] = torch.roll(Fg[ib], dt[t], 1)
    F0m = Fg.mean(0)

    return imin, yblk, F0, F0m


def segment_batches(ops, bfile):
    """Assign each sorted batch to a recording segment.

    A batch belongs to the segment containing its *first* sample. Batches that
    span a boundary are flagged; they mix data from two segments and are
    excluded from fingerprint pooling, but they still receive a shift.

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
        Must contain `drift_segment_starts`.
    bfile : kilosort.io.BinaryFiltered
        Wrapped file object for handling data.

    Returns
    -------
    seg_of_batch : np.ndarray
        Segment index of each batch, shape (Nbatches,), with values in
        [0, n_segments).
    straddles : np.ndarray
        Boolean mask of batches which span a segment boundary, shape
        (Nbatches,).
    n_segments : int
        Number of segments which contain data.

    """

    starts = np.asarray(ops['drift_segment_starts'], dtype='int64')

    # Batch `b` (sorted index) covers raw samples
    #   [imin + b*NT*ds, imin + (b+1)*NT*ds)
    # see io.BinaryRWFile._get_batch_edges and padded_batch_to_torch, which does
    # `ibatch *= self.batch_downsampling`.
    NT = np.int64(bfile.NT) * np.int64(bfile.batch_downsampling)
    imin = np.int64(bfile.imin)
    nb = np.int64(bfile.n_batches)

    first = imin + np.arange(nb, dtype='int64') * NT
    last = first + NT - 1

    seg_first = np.searchsorted(starts, first, side='right') - 1
    seg_last = np.searchsorted(starts, last, side='right') - 1
    straddles = seg_first != seg_last

    # tmin/tmax may crop whole segments away; drop empty ones and relabel 0..k-1
    used, seg_of_batch = np.unique(seg_first, return_inverse=True)
    seg_of_batch = seg_of_batch.reshape(-1).astype('int64')
    n_segments = used.size
    ops['drift_segment_used'] = used            # original indices, for reporting
    ops['drift_segment_starts_used'] = starts[used]

    if n_segments < 2:
        raise ValueError(
            f'Chronic drift mode needs at least 2 segments within '
            f'[tmin, tmax], but only {n_segments} of '
            f'{starts.size} segment(s) contain data. Check '
            '`drift_segment_starts`, `tmin`, and `tmax`.'
            )

    counts = np.bincount(seg_of_batch, minlength=n_segments)
    logger.info(f'Chronic drift mode: {n_segments} segments, '
                f'{counts.min()}-{counts.max()} batches each '
                f'({straddles.sum()} boundary-straddling batches excluded '
                'from fingerprints).')

    return seg_of_batch, straddles, n_segments


def segment_residual_drift(ops, st, seg_of_batch, straddles, F_seg, ysamp,
                           device=torch.device('cuda'), max_batches=2000):
    """Estimate per-batch residual shift relative to each segment's fingerprint.

    This tests the assumption that chronic mode makes: if drift really is
    constant within a segment, then every batch of that segment should register
    to that segment's own pooled fingerprint at ~0 offset.

    Diagnostic only, this does not modify `dshift`. Stores in `ops`:
      'drift_residual'         : (n_sampled_batches, nblocks_eff) microns
      'drift_residual_batches' : batch indices the residuals correspond to
      'drift_residual_summary' : (n_segments, nblocks_eff, 4) array of
                                 [median, std, p5, p95] in microns

    Parameters
    ----------
    ops : dict
        Dictionary storing settings and results for all algorithmic steps.
    st : np.ndarray
        Spike times array with 6 columns, as returned by `spikedetect.run`.
    seg_of_batch : np.ndarray
        Segment index of each batch, from `segment_batches`.
    straddles : np.ndarray
        Boolean mask of boundary-straddling batches, from `segment_batches`.
        Those batches mix two segments, so their residual would be a
        meaningless outlier and they are skipped.
    F_seg : np.ndarray
        Pooled per-segment fingerprints, shape (n_segments, dmax, 20).
    ysamp : np.ndarray
        Center of each vertical sampling bin.
    device : torch.device; default=torch.device('cuda').
    max_batches : int; default=2000.
        Maximum number of batches evaluated per segment. Larger segments are
        subsampled evenly, so that memory and runtime stay bounded.

    Returns
    -------
    ops : dict

    """

    n_seg = F_seg.shape[0]
    dd = ops['binning_depth']

    # Same block boundaries as `align_block2`, so that the residuals are
    # directly comparable to `dshift`.
    ifirst, ilast = block_indices(F_seg.shape[1], ops['nblocks'])
    nblocks_eff = len(ifirst)

    # Same restricted search and x10 upsampling as the fine step of align_block2
    n = 5
    dt = np.arange(-n, n+1, 1)
    dtup = np.linspace(-n, n, 2*n*10+1)
    Kn = kernelD(dt, dtup, 1)

    sp_batch = st[:,4].astype('int64')
    all_res = []
    all_batches = []
    summary = np.full((n_seg, nblocks_eff, 4), np.nan)

    # One segment at a time, so that peak memory is max_batches x dmax x 20
    # floats rather than Nbatches x dmax x 20. Doing it any other way would
    # reintroduce exactly the problem chronic mode is meant to solve.
    for s in range(n_seg):
        batches = np.flatnonzero((seg_of_batch == s) & ~straddles)
        if batches.size == 0:
            continue
        if batches.size > max_batches:
            batches = batches[::int(np.ceil(batches.size/max_batches))]

        # Relabel the selected batches to a local contiguous index. `sp_batch`
        # is sorted ascending and the relabeling is monotonic, so the resulting
        # group ids are sorted as `bin_spikes` requires.
        local = np.full(seg_of_batch.size, -1, dtype='int64')
        local[batches] = np.arange(batches.size)
        lid = local[sp_batch]
        keep = lid >= 0
        Fb, _ = bin_spikes(ops, st[keep], group_id=lid[keep],
                           n_groups=batches.size)

        # mean subtract along depth, as align_block2 does
        Fb = torch.from_numpy(Fb).to(device).float()
        Fb = Fb - Fb.mean(1).unsqueeze(1)

        # The template is this segment's own fingerprint, and the segment's own
        # shift is deliberately *not* applied: the residual is measured in the
        # segment's own frame, which sidesteps any sign convention question.
        F0 = torch.from_numpy(F_seg[s]).to(device).float()
        F0 = F0 - F0.mean(0).unsqueeze(0)

        dcs = np.zeros((2*n+1, batches.size, nblocks_eff))
        for j in range(nblocks_eff):
            isub = np.arange(ifirst[j], ilast[j], 1)
            Fsub = Fb[:, isub]
            for t in range(len(dt)):
                Fs = torch.roll(Fsub, dt[t], 1)
                dcs[t,:,j] = (Fs * F0[isub]).mean(-1).mean(-1).cpu().numpy()

        res = np.zeros((batches.size, nblocks_eff))
        for j in range(nblocks_eff):
            dcup = Kn.T @ dcs[:,:,j]
            res[:,j] = dtup[np.argmax(dcup, 0)] * dd

        summary[s,:,0] = np.median(res, axis=0)
        summary[s,:,1] = np.std(res, axis=0)
        summary[s,:,2] = np.percentile(res, 5, axis=0)
        summary[s,:,3] = np.percentile(res, 95, axis=0)

        all_res.append(res)
        all_batches.append(batches)

    if len(all_res) == 0:
        logger.warning('No batches available for residual drift diagnostics.')
        return ops

    ops['drift_residual'] = np.concatenate(all_res, axis=0)
    ops['drift_residual_batches'] = np.concatenate(all_batches, axis=0)
    ops['drift_residual_summary'] = summary

    _log_residual_summary(ops, summary, seg_of_batch, straddles)

    return ops


def _log_residual_summary(ops, summary, seg_of_batch, straddles):
    """Log the residual drift table, and warn if the assumption is violated."""

    # A residual spread below the binning resolution cannot be distinguished
    # from estimation noise, so use that (or half the interpolation scale,
    # whichever is larger) as the tolerance.
    tol = max(ops['binning_depth'], 0.5*ops['sig_interp'])
    n_seg = summary.shape[0]
    used = ops.get('drift_segment_used', np.arange(n_seg))

    logger.info('Within-segment residual drift (worst block per segment):')
    logger.info('  seg  batches  median(um)       p5-p95(um)  max|res|(um)')
    spread_bad = []
    bias_bad = []
    for s in range(n_seg):
        if np.all(np.isnan(summary[s])):
            continue
        nb = int(np.sum((seg_of_batch == s) & ~straddles))
        spread = summary[s,:,3] - summary[s,:,2]
        j = int(np.argmax(spread))
        med, p5, p95 = summary[s,j,0], summary[s,j,2], summary[s,j,3]
        mx = np.max(np.abs(summary[s][:,[0,2,3]]))
        logger.info(f'  {used[s]:>3}  {nb:>7}  {med:>+10.2f}  '
                    f'{p5:>+6.2f}..{p95:>+6.2f}  {mx:>12.2f}')
        if spread[j] > tol:
            spread_bad.append((int(used[s]), spread[j]))
        if np.max(np.abs(summary[s,:,0])) > tol:
            bias_bad.append((int(used[s]), np.max(np.abs(summary[s,:,0]))))

    if len(spread_bad) > 0:
        seg_txt = ', '.join([f'{s} ({v:.1f} um)' for s, v in spread_bad])
        warnings.warn(
            f'Within-segment residual drift exceeds {tol:.1f} um (p5-p95) for '
            f'segment(s): {seg_txt}. Drift is not constant within those '
            'segments, so the chronic model is a poor fit for them. Either '
            'split them into shorter segments, or use standard per-batch '
            'drift correction (`drift_segment_starts = None`).',
            UserWarning
            )

    if len(bias_bad) > 0:
        seg_txt = ', '.join([f'{s} ({v:+.1f} um)' for s, v in bias_bad])
        warnings.warn(
            f'Median within-segment residual exceeds {tol:.1f} um for '
            f'segment(s): {seg_txt}. A consistently non-zero median, as '
            'opposed to a large spread, means the pooled registration is '
            'biased for those segments. That usually indicates that the '
            'fingerprint content changed rather than just its position '
            '(units lost or gained, or a gain change). Inspect '
            'drift_segments.png before trusting across-segment unit identity.',
            UserWarning
            )


def kernelD(x, y, sig = 1):
    ds = (x[:,np.newaxis] - y)
    Kn = np.exp(-ds**2 / (2*sig**2))
    return Kn

def kernel2D_torch(x, y, sig = 1):
    ds = ((x.unsqueeze(1) - y)**2).sum(-1)
    Kn = torch.exp(-ds / (2*sig**2))
    return Kn

def kernel2D(x, y, sig = 1):
    ds = ((x[:,np.newaxis] - y)**2).sum(-1)
    Kn = np.exp(-ds / (2*sig**2))
    return Kn

def run(ops, bfile, device=torch.device('cuda'), progress_bar=None,
        clear_cache=False, verbose=False):
    """ this step computes a drift correction model
    it returns vertical correction amplitudes for each batch, and for multiple blocks in a batch if nblocks > 1.
    """

    if ops['nblocks']<1:
        ops['dshift'] = None
        ops['batch_to_segment'] = None
        logger.info('nblocks = 0, skipping drift correction')
        return ops, None

    # the first step is to extract all spikes using the universal templates
    st, _, ops  = spikedetect.run(
        ops, bfile, device=device, progress_bar=progress_bar,
        clear_cache=clear_cache, verbose=verbose
        )

    segment_starts = ops.get('drift_segment_starts', None)
    if segment_starts is None:
        # --- standard per-batch estimation ---
        # spikes are binned by amplitude and y-position to construct a
        # "fingerprint" for each batch
        F, ysamp = bin_spikes(ops, st)

        # the fingerprints are iteratively aligned to each other vertically
        imin, yblk, _, _ = align_block2(F, ysamp, ops, device=device)
        ops['batch_to_segment'] = None
    else:
        # --- chronic mode: one shift per (segment, block) ---
        seg_of_batch, straddles, n_seg = segment_batches(ops, bfile)
        sp_batch = st[:,4].astype('int64')
        keep = ~straddles[sp_batch]
        gid = seg_of_batch[sp_batch[keep]]        # already sorted ascending

        # all spikes of a segment are pooled into a single fingerprint
        F, ysamp = bin_spikes(ops, st[keep], group_id=gid, n_groups=n_seg)

        # Never smooth the dot products across segment boundaries: a
        # segment-to-segment step is real signal, not noise. The axis order is
        # (correlation, time, block).
        smoothing = list(ops['drift_smoothing'])
        smoothing[1] = 0.0

        imin_seg, yblk, _, _ = align_block2(
            F, ysamp, ops, device=device, drift_smoothing=smoothing
            )

        # Broadcast the per-segment shift back to every batch. `dshift` keeps
        # its original (Nbatches, nblocks_eff) shape, so nothing downstream
        # needs to know that chronic mode was used.
        imin = imin_seg[seg_of_batch]
        ops['batch_to_segment'] = seg_of_batch
        ops['drift_segment_shift'] = imin_seg * ops['binning_depth']
        ops['drift_segment_fingerprints'] = F

        if ops.get('drift_segment_diagnostics', True):
            ops = segment_residual_drift(ops, st, seg_of_batch, straddles, F,
                                         ysamp, device=device)

    # imin contains the shifts for each batch, in units of discrete bins
    # multiply back with binning_depth for microns
    dshift = imin * ops['binning_depth']

    # we save the variables needed for drift correction during the data preprocessing step
    ops['yblk'] = yblk
    ops['dshift'] = dshift
    xp = np.vstack((ops['xc'],ops['yc'])).T

    # for interpolation, we precompute a radial kernel based on distances between sites
    Kxx = kernel2D(xp, xp, ops['sig_interp'])
    Kxx = torch.from_numpy(Kxx).to(device)

    # a small constant is added to the diagonal for stability of the matrix inversion
    ops['iKxx'] = torch.linalg.inv(Kxx + 0.01 * torch.eye(Kxx.shape[0], device=device))

    return ops, st
