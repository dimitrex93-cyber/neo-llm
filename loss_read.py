from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import glob, datetime

runs = sorted(glob.glob('out/runs/Aug27_17-*'))
if not runs:
    print('KEIN Run gefunden')
else:
    run = runs[-1]
    print('Run:', run.split('/')[-1])
    events = []
    for path in glob.glob(run + '/*.tfevents*'):
        loader = EventFileLoader(path)
        for ev in loader.Load():
            if ev.summary and ev.summary.value:
                for v in ev.summary.value:
                    if v.tag in ('train/loss', 'train/learning_rate', 'train/epoch'):
                        val = v.tensor.float_val[0] if v.tensor and v.tensor.float_val else None
                        if val is not None:
                            events.append((v.tag, ev.step, val))
    losses = [(s, v) for t, s, v in events if t == 'train/loss']
    lrs = [(s, v) for t, s, v in events if t == 'train/learning_rate']
    epochs = [(s, v) for t, s, v in events if t == 'train/epoch']
    if losses:
        print(f'Steps geloggt: {losses[-1][0]} (bis jetzt)')
        # Letzte 10
        for s, v in losses[-10:]:
            print(f'  Step {s}: loss={v:.4f}')
        print('Bester Loss:', min(v for _, v in losses))
        # Rate aus Zeitstempeln
        import os
        times = []
        for path in glob.glob(run + '/*.tfevents*'):
            loader = EventFileLoader(path)
            for ev in loader.Load():
                if ev.summary and ev.summary.value:
                    for v in ev.summary.value:
                        if v.tag == 'train/loss':
                            times.append((ev.step, ev.wall_time))
        if len(times) >= 2:
            dt = times[-1][1] - times[0][1]
            ds = times[-1][0] - times[0][0]
            if ds > 0:
                rate = dt / ds
                rest = (1580 - times[-1][0]) * rate
                print(f'Rate: {rate:.1f} s/Step | Rest: ~{rest/3600:.1f} h')
    if lrs:
        print('LR (letzter):', lrs[-1][1])
    if epochs:
        print('Epoch (letzter):', epochs[-1][1])
