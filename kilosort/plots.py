import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


COLOR_CODES = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
# This is matplotlib tab10 with gray moved to the last position and all 0.5 alpha
PROBE_PLOT_COLORS = np.array([
        [0.12156863, 0.46666667, 0.70588235, 0.5],
        [1.        , 0.49803922, 0.05490196, 0.5],
        [0.17254902, 0.62745098, 0.17254902, 0.5],
        [0.83921569, 0.15294118, 0.15686275, 0.5],
        [0.58039216, 0.40392157, 0.74117647, 0.5],
        [0.54901961, 0.3372549 , 0.29411765, 0.5],
        [0.89019608, 0.46666667, 0.76078431, 0.5],
        [0.7372549 , 0.74117647, 0.13333333, 0.5],
        [0.09019608, 0.74509804, 0.81176471, 0.5],
        [0.49803922, 0.49803922, 0.49803922, 0.25]
    ])


def segment_boundary_times(ops, tmin=0):
    """Time (in seconds) of the first batch of each recording segment.

    Returns None if chronic drift correction was not used. The times are
    computed from batch indices rather than from sample indices, so that they
    line up exactly with the time axis used by the drift plots.

    """
    seg = ops.get('batch_to_segment', None)
    if seg is None:
        return None

    settings = ops['settings']
    fs = settings['fs']
    NT = settings['batch_size']
    # first batch of each segment, excluding the first segment (t = tmin)
    ibatch = np.flatnonzero(np.diff(seg)) + 1

    return ibatch*(NT/fs) + tmin


def plot_drift_amount(ops, results_dir, tmin=0):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    dshift = ops['dshift']
    settings = ops['settings']

    fs = settings['fs']
    NT = settings['batch_size']
    t = np.arange(dshift.shape[0])*(NT/fs) + tmin
    for i in range(dshift.shape[1]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        ax.plot(t, dshift[:,i], c=color)

    boundaries = segment_boundary_times(ops, tmin=tmin)
    if boundaries is not None:
        for b in boundaries:
            if t[0] <= b <= t[-1]:
                ax.axvline(b, c='gray', ls='--', lw=0.75)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth shift (um)')
    fig.suptitle('Drift amount per probe section, across batches')
    fig.tight_layout()

    save_path = results_dir / 'drift_amount.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def plot_chronic_drift(ops, results_dir, tmin=0):
    """Plot per-segment drift, the within-segment residual, and learned shapes.

    Only meaningful when `drift_segment_starts` was set. The top panel shows
    the drift estimate, the middle panel the residual per-batch shift within
    each segment (after the learned shape, if one was fit), which is the check
    on the within-segment model. If `drift_shape_rank > 0`, the bottom panel
    shows the learned shapes over the course of a segment.

    """
    if ops.get('batch_to_segment', None) is None:
        return

    dshift = ops['dshift']
    settings = ops['settings']
    fs = settings['fs']
    NT = settings['batch_size']
    dd = ops['binning_depth']
    t = np.arange(dshift.shape[0])*(NT/fs) + tmin
    boundaries = segment_boundary_times(ops, tmin=tmin)
    residual = ops.get('drift_residual', None)
    curves = ops.get('drift_shape_curves', None)
    has_shape = curves is not None

    plt.style.use('dark_background')
    nrows = 1 + (residual is not None) + has_shape
    fig = plt.figure(figsize=(10, 4*nrows + 1))

    ax = fig.add_subplot(nrows, 1, 1)
    time_axes = [ax]
    for i in range(dshift.shape[1]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        ax.step(t, dshift[:,i], where='post', c=color, lw=1)
    ax.set_ylabel('Depth shift (um)')
    title = 'Drift per segment' + (' with learned shape' if has_shape else '')
    ax.set_title(f'{title} ({ops["drift_segment_used"].size} segments, '
                 f'{dshift.shape[1]} blocks)')

    if residual is not None:
        ax = fig.add_subplot(nrows, 1, 2, sharex=time_axes[0])
        time_axes.append(ax)
        tr = ops['drift_residual_batches']*(NT/fs) + tmin
        ax.axhspan(-dd, dd, color='gray', alpha=0.3, zorder=0)
        for i in range(residual.shape[1]):
            color = COLOR_CODES[i % len(COLOR_CODES)]
            ax.plot(tr, residual[:,i], c=color, lw=0.5, alpha=0.8)
        ax.set_ylabel('Residual shift (um)')
        after = ' after learned shape' if has_shape else ''
        ax.set_title(f'Within-segment residual drift{after} '
                     f'(shaded band: +/- binning_depth = {dd} um)')

    for ax in time_axes:
        if boundaries is not None:
            for b in boundaries:
                if t[0] <= b <= t[-1]:
                    ax.axvline(b, c='gray', ls='--', lw=0.75)
    time_axes[-1].set_xlabel('Time (s)')

    if has_shape:
        ax = fig.add_subplot(nrows, 1, nrows)
        grid = ops['drift_shape_time_grid']
        for k in range(curves.shape[1]):
            color = COLOR_CODES[k % len(COLOR_CODES)]
            ax.plot(grid, curves[:,k], c=color, label=f'shape {k}')
        ax.axhline(0, c='gray', lw=0.5)
        ax.set_xlabel('Position within segment (0 = first batch, 1 = last)')
        ax.set_ylabel('Shape (unit RMS)')
        ax.set_title('Learned within-segment drift shape '
                     '(amount per segment: ops["drift_shape_amplitude"])')
        if curves.shape[1] > 1:
            ax.legend()

    fig.suptitle('Chronic drift correction')
    fig.tight_layout()

    save_path = results_dir / 'drift_segments.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def _amplitude_colors(amp):
    """Map spike amplitudes to grayscale colors, binned on a log scale."""
    # np.clip copies, so `st0` itself is left untouched
    z = np.clip(amp, 10, 100)
    colors = np.empty((z.shape[0], 4), dtype=float)

    bin_idx = np.digitize(z, np.logspace(1, 2, 90))
    cm = matplotlib.colormaps['binary']
    for i in np.unique(bin_idx):
        # Take mean of all amplitude values within one bin, map to color
        subset = (bin_idx == i)
        a = z[subset].mean()
        colors[subset] = cm(((a-10)/90))

    return colors


def _save_depth_scatter(t, y, amp, title, save_path, boundaries=None):
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    # Scatter of spike depth over time, with color intensity proportional
    # to log of amplitude.
    ax.scatter(t, y, s=3, c=_amplitude_colors(amp))
    if boundaries is not None:
        for b in boundaries:
            ax.axvline(b, c='tab:red', ls='--', lw=1)
    ax.set_xlabel('Time (s)', fontsize=22)
    ax.set_ylabel('Depth (um)', fontsize=22)
    fig.suptitle(title, fontsize=30)
    fig.tight_layout()

    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_drift_scatter(st0, results_dir, tmin=0):
    x = st0[:,0] + tmin  # spike time in seconds
    y = st0[:,1]         # depth of spike center in microns
    z = st0[:,2]         # spike amplitude (data)
    _save_depth_scatter(x, y, z, 'Spike amplitude across time and depth',
                        results_dir / 'drift_scatter.png')


def corrected_spike_depths(st0, ops):
    """Depth of each drift-detection spike after applying `ops['dshift']`.

    `preprocessing.get_drift_matrix` makes a channel at depth `yc` read the
    raw data at `yc - shift`, so a spike detected at raw depth `y` ends up at
    `y + shift`. With multiple blocks the shift is interpolated linearly over
    `ops['yblk']` (extrapolated past the ends), as it is for the channels.

    """
    y = st0[:,1]
    batch = st0[:,4].astype('int64')
    dshift = ops['dshift']
    if dshift.shape[1] == 1:
        shift = dshift[batch, 0]
    else:
        yblk = ops['yblk']
        # position of each spike between neighbouring block centers
        j = np.clip(np.searchsorted(yblk, y) - 1, 0, yblk.size - 2)
        w = (y - yblk[j]) / (yblk[j+1] - yblk[j])
        shift = (1 - w)*dshift[batch, j] + w*dshift[batch, j+1]

    return y + shift


def plot_drift_scatter_corrected(st0, ops, results_dir, tmin=0):
    """Same as `plot_drift_scatter`, with spike depths drift-corrected.

    If the correction worked, units appear as flat horizontal bands. In
    chronic mode, segment boundaries are marked, since remaining jumps there
    point to a misestimated per-segment shift.

    """
    x = st0[:,0] + tmin
    y = corrected_spike_depths(st0, ops)
    z = st0[:,2]
    _save_depth_scatter(
        x, y, z, 'Spike amplitude across time and depth, after drift correction',
        results_dir / 'drift_scatter_corrected.png',
        boundaries=segment_boundary_times(ops, tmin=tmin)
        )


def plot_diagnostics(Wall0, clu0, ops, results_dir):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16,16))
    wPCA = ops['wPCA']
    settings = ops['settings']

    # Top left
    t = np.arange(wPCA.shape[1])/(settings['fs']/1000)
    for i in range(wPCA.shape[0]):
        color = COLOR_CODES[i % len(COLOR_CODES)]
        axes[0][0].plot(t, wPCA[i,:].cpu().numpy(), c=color)
    axes[0][0].set_xlabel('Time (s)')
    axes[0][0].set_title('Temporal Features')

    # Top right
    features = torch.linalg.norm(Wall0, dim=2).cpu().numpy()
    axes[0][1].imshow(features.T, aspect='auto', vmin=0, vmax=25, cmap='binary_r')
    axes[0][1].set_xlabel('Channel Number')
    axes[0][1].set_ylabel('Unit Number')
    axes[0][1].set_title('Spatial Features')

    # Comput spike counts and mean amplitudes
    n_units = int(clu0.max()) + 1
    spike_counts = np.zeros(n_units)
    for i in range(n_units):
        spike_counts[i] = (clu0[clu0 == i]).size
    mean_amp = torch.linalg.norm(Wall0, dim=(1,2)).cpu().numpy()

    # Bottom left
    axes[1][0].plot(mean_amp)
    axes[1][0].set_xlabel('Unit Number')
    axes[1][0].set_ylabel('Amplitude (a.u.)')
    axes[1][0].set_title('Unit Amplitudes')

    # Bottom right
    axes[1][1].scatter(np.log(1 + spike_counts), mean_amp, s=3)
    axes[1][1].set_xlabel('Log(1 + Spike Count)')
    axes[1][1].set_ylabel('Amplitude (a.u.)')
    axes[1][1].set_title('Amplitude vs Spike Count')

    fig.tight_layout()
    save_path = results_dir / 'diagnostics.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)


def plot_spike_positions(clu, is_refractory, results_dir):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(30,14))

    # 10 colors in palette, last one is gray for non-frefractory
    clu = clu.copy()
    bad_units = np.unique(clu)[is_refractory == 0]
    bad_idx = np.isin(clu, bad_units)
    clu = np.mod(clu, 9)
    clu[bad_idx] = 9
    colors = np.empty((clu.shape[0], 4), dtype=float)

    # Map modded cluster ids to colors
    for i in range(10):
        subset = (clu == i)
        rgba = PROBE_PLOT_COLORS[i]
        colors[subset] = rgba

    # Get x, y positions, add to scatterplot
    positions = np.load(results_dir / 'spike_positions.npy')
    xs, ys = positions[:,0], positions[:,1]
    ax.scatter(ys, xs, s=3, c=colors)
    ax.set_xlabel('Depth (um)', fontsize=22)
    ax.set_ylabel('Lateral (um)', fontsize=22)
    fig.suptitle('Spike position across probe, colored by cluster', fontsize=30)
    fig.tight_layout()

    save_path = results_dir / 'spike_positions.png'
    fig.savefig(save_path, dpi=300)
    plt.style.use('default')
    plt.close(fig)
