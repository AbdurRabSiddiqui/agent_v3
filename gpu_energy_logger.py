from __future__ import annotations

import argparse
import csv
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
  import pynvml  # type: ignore[reportMissingImports]
  nvmlInit = pynvml.nvmlInit
  nvmlShutdown = pynvml.nvmlShutdown
  nvmlDeviceGetCount = pynvml.nvmlDeviceGetCount
  nvmlDeviceGetHandleByIndex = pynvml.nvmlDeviceGetHandleByIndex
  nvmlDeviceGetName = pynvml.nvmlDeviceGetName
  nvmlDeviceGetPowerUsage = pynvml.nvmlDeviceGetPowerUsage
except Exception:  # pynvml is optional until runtime
  nvmlInit = None


@dataclass(frozen=True)
class GpuSample:
  elapsed_s: float
  total_power_w: float
  effective_power_w: float
  total_energy_j: float
  effective_energy_j: float


def _require_nvml():
  if nvmlInit is None:
    raise RuntimeError(
      'pynvml is not installed. Install `nvidia-ml-py` (and ensure an NVIDIA GPU + driver are available).'
    )


def _to_text(value) -> str:
  if value is None:
    return ''
  if isinstance(value, bytes):
    return value.decode(errors='ignore')
  return str(value)


def list_gpus() -> list[tuple[int, str]]:
  _require_nvml()
  nvmlInit()
  try:
    count = nvmlDeviceGetCount()
    gpus: list[tuple[int, str]] = []
    for i in range(count):
      handle = nvmlDeviceGetHandleByIndex(i)
      name = _to_text(nvmlDeviceGetName(handle))
      gpus.append((i, name))
    return gpus
  finally:
    nvmlShutdown()


class GPUEnergyLogger:
  def __init__(
    self,
    *,
    device_index: int = 0,
    sample_interval_s: float = 0.2,
    baseline_duration_s: float = 5.0,
    csv_path: Optional[str] = None,
    flush_csv: bool = True,
    store_series: bool = True
  ):
    _require_nvml()
    self.device_index = device_index
    self.sample_interval_s = max(float(sample_interval_s), 0.05)
    self.baseline_duration_s = max(float(baseline_duration_s), 0.0)
    self.csv_path = csv_path
    self.flush_csv = flush_csv
    self.store_series = bool(store_series)

    self._running = False
    self._thread: Optional[threading.Thread] = None
    self._lock = threading.Lock()

    self._idle_power_w = 0.0
    self._effective_energy_j = 0.0
    self._total_energy_j = 0.0
    self._last_sample: Optional[GpuSample] = None

    self.timestamps: list[float] = []
    self.total_power_series: list[float] = []
    self.effective_power_series: list[float] = []
    self.total_energy_series: list[float] = []
    self.effective_energy_series: list[float] = []

    nvmlInit()
    try:
      device_count = nvmlDeviceGetCount()
      if device_index < 0 or device_index >= device_count:
        raise ValueError(f'GPU index {device_index} not available (count={device_count})')
      self.handle = nvmlDeviceGetHandleByIndex(device_index)
      self.device_name = _to_text(nvmlDeviceGetName(self.handle))
    except Exception:
      nvmlShutdown()
      raise

  def shutdown(self):
    try:
      self.stop()
    finally:
      nvmlShutdown()

  def reset(self):
    with self._lock:
      self._idle_power_w = 0.0
      self._effective_energy_j = 0.0
      self._total_energy_j = 0.0
      self._last_sample = None
      self.timestamps.clear()
      self.total_power_series.clear()
      self.effective_power_series.clear()
      self.total_energy_series.clear()
      self.effective_energy_series.clear()

  def calibrate_idle_baseline(self):
    if self.baseline_duration_s <= 0:
      self._idle_power_w = 0.0
      return

    print('Calibrating idle baseline... ensure GPU is idle.')
    samples: list[float] = []
    start = time.time()
    while time.time() - start < self.baseline_duration_s:
      power_mw = nvmlDeviceGetPowerUsage(self.handle)
      samples.append(power_mw / 1000.0)
      time.sleep(self.sample_interval_s)

    self._idle_power_w = (sum(samples) / len(samples)) if samples else 0.0
    print(f'Idle baseline: {self._idle_power_w:.2f} W')

  def is_running(self) -> bool:
    return self._running

  def snapshot(self) -> Optional[GpuSample]:
    with self._lock:
      if self._last_sample is None:
        return None
      return self._last_sample

  def _monitor_loop(self):
    last_time = time.time()
    start_time = last_time

    csv_file = None
    csv_writer = None

    try:
      if self.csv_path:
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(self.csv_path, mode='w', newline='', buffering=1)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
          'elapsed_s',
          'total_power_w',
          'effective_power_w',
          'total_energy_j',
          'effective_energy_j'
        ])
        if self.flush_csv:
          csv_file.flush()

      while self._running:
        now = time.time()
        dt = max(now - last_time, 0.0)
        elapsed = now - start_time
        last_time = now

        power_mw = nvmlDeviceGetPowerUsage(self.handle)
        total_power = power_mw / 1000.0
        effective_power = max(total_power - self._idle_power_w, 0.0)

        with self._lock:
          self._total_energy_j += total_power * dt
          self._effective_energy_j += effective_power * dt

          sample = GpuSample(
            elapsed_s=elapsed,
            total_power_w=total_power,
            effective_power_w=effective_power,
            total_energy_j=self._total_energy_j,
            effective_energy_j=self._effective_energy_j
          )
          self._last_sample = sample

          if self.store_series:
            self.timestamps.append(elapsed)
            self.total_power_series.append(total_power)
            self.effective_power_series.append(effective_power)
            self.total_energy_series.append(self._total_energy_j)
            self.effective_energy_series.append(self._effective_energy_j)

        if csv_writer:
          csv_writer.writerow([
            sample.elapsed_s,
            sample.total_power_w,
            sample.effective_power_w,
            sample.total_energy_j,
            sample.effective_energy_j
          ])
          if self.flush_csv:
            csv_file.flush()

        time.sleep(self.sample_interval_s)
    finally:
      if csv_file:
        csv_file.close()

  def start(self):
    if self._running:
      return
    self._running = True
    self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
    self._thread.start()

  def stop(self) -> Optional[GpuSample]:
    if not self._running:
      return self.snapshot()
    self._running = False
    if self._thread:
      self._thread.join(timeout=5)
    return self.snapshot()

  def save_plots(self, *, out_dir: str, title_prefix: str = '') -> tuple[str, str]:
    """
    Save power/energy graphs as PNGs into out_dir.
    Returns (power_png_path, energy_png_path).
    """
    try:
      import matplotlib.pyplot as plt
    except Exception as e:
      raise RuntimeError('matplotlib is required for graphs. Install `matplotlib`.') from e

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with self._lock:
      ts = list(self.timestamps)
      total_p = list(self.total_power_series)
      eff_p = list(self.effective_power_series)
      total_e = list(self.total_energy_series)
      eff_e = list(self.effective_energy_series)

    if not ts:
      raise RuntimeError('No samples recorded; cannot plot.')

    power_png = str(Path(out_dir) / 'gpu_power.png')
    energy_png = str(Path(out_dir) / 'gpu_energy.png')

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(ts, total_p, label='total_power_w')
    ax1.plot(ts, eff_p, label='effective_power_w')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Power (W)')
    ax1.set_title((title_prefix + ' ' if title_prefix else '') + 'GPU power vs time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(power_png, dpi=180)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(ts, total_e, label='total_energy_j')
    ax2.plot(ts, eff_e, label='effective_energy_j')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (J)')
    ax2.set_title((title_prefix + ' ' if title_prefix else '') + 'GPU energy vs time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(energy_png, dpi=180)
    plt.close(fig2)

    return power_png, energy_png


def _default_csv_path() -> str:
  ts = datetime.now().strftime('%Y%m%d_%H%M%S')
  return str(Path('logs') / f'gpu_energy_{ts}.csv')


def save_plots_from_csv(*, csv_path: str, out_dir: str, title_prefix: str = '') -> tuple[str, str]:
  try:
    import matplotlib.pyplot as plt
  except Exception as e:
    raise RuntimeError('matplotlib is required for graphs. Install `matplotlib`.') from e

  if not csv_path:
    raise RuntimeError('csv_path is required')

  ts: list[float] = []
  total_p: list[float] = []
  eff_p: list[float] = []
  total_e: list[float] = []
  eff_e: list[float] = []

  with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
      ts.append(float(row['elapsed_s']))
      total_p.append(float(row['total_power_w']))
      eff_p.append(float(row['effective_power_w']))
      total_e.append(float(row['total_energy_j']))
      eff_e.append(float(row['effective_energy_j']))

  if not ts:
    raise RuntimeError('CSV has no samples; cannot plot.')

  Path(out_dir).mkdir(parents=True, exist_ok=True)
  power_png = str(Path(out_dir) / 'gpu_power.png')
  energy_png = str(Path(out_dir) / 'gpu_energy.png')

  fig1, ax1 = plt.subplots(figsize=(10, 4))
  ax1.plot(ts, total_p, label='total_power_w')
  ax1.plot(ts, eff_p, label='effective_power_w')
  ax1.set_xlabel('Time (s)')
  ax1.set_ylabel('Power (W)')
  ax1.set_title((title_prefix + ' ' if title_prefix else '') + 'GPU power vs time')
  ax1.grid(True, alpha=0.3)
  ax1.legend()
  fig1.tight_layout()
  fig1.savefig(power_png, dpi=180)
  plt.close(fig1)

  fig2, ax2 = plt.subplots(figsize=(10, 4))
  ax2.plot(ts, total_e, label='total_energy_j')
  ax2.plot(ts, eff_e, label='effective_energy_j')
  ax2.set_xlabel('Time (s)')
  ax2.set_ylabel('Energy (J)')
  ax2.set_title((title_prefix + ' ' if title_prefix else '') + 'GPU energy vs time')
  ax2.grid(True, alpha=0.3)
  ax2.legend()
  fig2.tight_layout()
  fig2.savefig(energy_png, dpi=180)
  plt.close(fig2)

  return power_png, energy_png


def _run_live_plot(logger: GPUEnergyLogger, *, refresh_ms: int = 200):
  try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
  except Exception as e:
    raise RuntimeError('matplotlib is required for live graphs. Install `matplotlib`.') from e

  fig, (ax_power, ax_energy) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

  (total_power_line,) = ax_power.plot([], [], label='total_power_w')
  (effective_power_line,) = ax_power.plot([], [], label='effective_power_w')
  ax_power.set_ylabel('Power (W)')
  ax_power.set_title(f'GPU power ({logger.device_name})')
  ax_power.grid(True, alpha=0.3)
  ax_power.legend(loc='upper right')

  (total_energy_line,) = ax_energy.plot([], [], label='total_energy_j')
  (effective_energy_line,) = ax_energy.plot([], [], label='effective_energy_j')
  ax_energy.set_xlabel('Time (s)')
  ax_energy.set_ylabel('Energy (J)')
  ax_energy.set_title('GPU energy')
  ax_energy.grid(True, alpha=0.3)
  ax_energy.legend(loc='upper right')

  def update(_frame):
    with logger._lock:
      if not logger.timestamps:
        return []

      ts = logger.timestamps
      total_p = logger.total_power_series
      eff_p = logger.effective_power_series
      total_e = logger.total_energy_series
      eff_e = logger.effective_energy_series

    total_power_line.set_data(ts, total_p)
    effective_power_line.set_data(ts, eff_p)
    total_energy_line.set_data(ts, total_e)
    effective_energy_line.set_data(ts, eff_e)

    ax_power.set_xlim(0, max(ts) if ts else 1)
    ax_power.set_ylim(0, (max(total_p) * 1.1) if total_p else 1)
    ax_energy.set_xlim(0, max(ts) if ts else 1)
    ax_energy.set_ylim(0, (max(total_e) * 1.1) if total_e else 1)

    return [total_power_line, effective_power_line, total_energy_line, effective_energy_line]

  def on_close(_event):
    logger.stop()

  fig.canvas.mpl_connect('close_event', on_close)
  FuncAnimation(fig, update, interval=refresh_ms)
  plt.tight_layout()
  plt.show()


def main() -> int:
  parser = argparse.ArgumentParser(description='Log live GPU power/energy to CSV and graphs (NVIDIA NVML).')
  parser.add_argument('--list', action='store_true', help='List available NVIDIA GPUs and exit')
  parser.add_argument('--gpu', type=int, default=0, help='GPU index (default: 0)')
  parser.add_argument('--interval', type=float, default=0.2, help='Sample interval in seconds (default: 0.2)')
  parser.add_argument('--baseline', type=float, default=5.0, help='Idle baseline duration in seconds (default: 5)')
  parser.add_argument('--no-baseline', action='store_true', help='Disable idle baseline subtraction')
  parser.add_argument('--duration', type=float, default=0.0, help='Run for N seconds (0 = until Ctrl+C / plot close)')
  parser.add_argument('--csv', default=None, help='CSV output path (default: logs/gpu_energy_<ts>.csv)')
  parser.add_argument('--live-plot', action='store_true', help='Show live graphs while recording')
  parser.add_argument('--save-plots', action='store_true', help='Save PNG graphs at end (into --out-dir)')
  parser.add_argument('--out-dir', default='logs', help='Directory for saved PNGs (default: logs)')
  parser.add_argument('--print-every', type=float, default=2.0, help='Print a status line every N seconds (0=off)')
  parser.add_argument('--efficient', action='store_true', help='Minimize overhead (no in-memory series; plots generated from CSV)')
  args = parser.parse_args()

  if args.list:
    for idx, name in list_gpus():
      print(f'{idx}: {name}')
    return 0

  csv_path = args.csv or _default_csv_path()

  store_series = True
  if args.efficient:
    store_series = False
  if args.live_plot:
    store_series = True

  logger = GPUEnergyLogger(
    device_index=args.gpu,
    sample_interval_s=args.interval,
    baseline_duration_s=0.0 if args.no_baseline else args.baseline,
    csv_path=csv_path,
    store_series=store_series
  )

  exit_code = 0

  try:
    if not args.no_baseline and args.baseline > 0:
      logger.calibrate_idle_baseline()

    logger.start()
    print(f'[gpu] recording started: gpu={args.gpu} name="{logger.device_name}" csv="{csv_path}"')

    next_print = time.time()

    if args.live_plot:
      _run_live_plot(logger)
    else:
      start = time.time()
      while logger.is_running():
        if args.duration and (time.time() - start) >= args.duration:
          break
        if args.print_every and args.print_every > 0 and time.time() >= next_print:
          s = logger.snapshot()
          if s:
            print(
              f'[gpu] t={s.elapsed_s:7.1f}s  p={s.total_power_w:6.1f}W  '
              f'eff_p={s.effective_power_w:6.1f}W  e={s.total_energy_j:10.1f}J'
            )
          next_print = time.time() + args.print_every
        time.sleep(0.1)

    logger.stop()
    print('[gpu] recording stopped')

  except KeyboardInterrupt:
    logger.stop()
    print('\n[gpu] interrupted; recording stopped')
    exit_code = 130
  finally:
    if args.save_plots:
      if args.efficient:
        power_png, energy_png = save_plots_from_csv(
          csv_path=csv_path,
          out_dir=args.out_dir,
          title_prefix=logger.device_name
        )
      else:
        power_png, energy_png = logger.save_plots(out_dir=args.out_dir, title_prefix=logger.device_name)
      print(f'[gpu] saved plots: "{power_png}", "{energy_png}"')
    logger.shutdown()

  return exit_code


if __name__ == '__main__':
  raise SystemExit(main())
