"""Resume an offline BAFCv3 study with one isolated worker per selected GPU.

This scheduler leaves the evaluator and its source identity unchanged, so results
from an earlier sequential invocation remain valid. Example::

    python -m alf.bin.run_bafcv3_checkpoint_sweep --output STUDY --gpus 0 1 2 3
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import os
from pathlib import Path
import queue
import subprocess
import sys

from alf.bin.evaluate_bafcv3_checkpoints import (
    REPO, atomic_json, discover, read_json, source_identity, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)):
        raise ValueError('GPU IDs must be distinct')
    output = Path(args.output).resolve()
    manifest = read_json(output / 'manifest.json')
    if source_identity() != manifest['code']:
        raise ValueError('Evaluator code changed; use a new study or explicitly regenerate its results')
    manifest['parallel_execution'] = {
        'gpus': args.gpus, 'scheduler_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'note': 'One worker per physical GPU; child CUDA_VISIBLE_DEVICES maps it to logical cuda:0'}
    gpu_ids = queue.Queue()
    for gpu in args.gpus:
        gpu_ids.put(gpu)
    completed = set()
    filters = manifest.get('filters', {})

    def execute(request, log):
        gpu = gpu_ids.get()
        try:
            req = read_json(request)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            # Record assignment outside raw results so their evaluator fingerprint stays unchanged.
            assignment = {'physical_gpu': gpu, 'CUDA_VISIBLE_DEVICES': str(gpu),
                          'request': str(request), 'checkpoint': req['checkpoint']['path']}
            atomic_json(output / 'assignments' / request.name, assignment)
            with log.open('a') as stream:
                process = subprocess.run([sys.executable, '-m', 'alf.bin.evaluate_bafcv3_checkpoints',
                                          '--worker-request', str(request)], cwd=REPO, env=env,
                                         stdout=stream, stderr=subprocess.STDOUT)
            result = read_json(req['result']) if Path(req['result']).exists() else {'status': 'error'}
            return result.get('status', 'error'), result.get('rank_errors', {}), gpu, process.returncode
        finally:
            gpu_ids.put(gpu)

    for scan in range(2):
        runs = discover(manifest['source_glob'], filters.get('tasks', []),
                        filters.get('seeds', []), filters.get('checkpoints', []))
        manifest['runs'] = runs
        jobs = {j['checkpoint']: j for j in manifest['jobs']}
        planned = []
        for run in runs:
            expected = sorted({int(rank) for cp in run['checkpoints'] for rank in cp['shards']})
            for cp in run['checkpoints']:
                signature = (cp['path'], tuple((str(p), Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
                                              for p in [cp['path']] + list(cp['shards'].values())))
                ident = hashlib.sha256(cp['path'].encode()).hexdigest()[:16]
                item = jobs.setdefault(cp['path'], {'run': run['path'], 'checkpoint': cp['path'],
                                                   'result': f'checkpoints/{ident}.json.gz'})
                item['expected_ranks'] = expected
                item['missing_ranks'] = sorted(set(expected) - set(map(int, cp['shards'])))
                if signature in completed:
                    continue
                if not cp['shards']:
                    item['status'] = 'pending_no_replay'
                    continue
                request = {'run': run, 'checkpoint': cp, 'options': manifest['options'],
                           'code': manifest['code'], 'resume': True, 'result': str(output / item['result'])}
                path = output / 'requests' / f'{ident}.json'
                atomic_json(path, request)
                log = output / 'logs' / f'{ident}.log'
                log.parent.mkdir(exist_ok=True)
                item['status'] = 'queued'
                planned.append((path, log, item, signature))
        manifest['jobs'] = list(jobs.values())
        atomic_json(output / 'manifest.json', manifest)
        print('PARALLEL_SCAN', scan + 1, 'jobs', len(planned), flush=True)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            pending = {pool.submit(execute, req, log): (item, signature)
                       for req, log, item, signature in planned}
            for future in as_completed(pending):
                item, signature = pending[future]
                try:
                    status, errors, gpu, returncode = future.result()
                    item.update(status=status, rank_errors=errors, physical_gpu=gpu, returncode=returncode)
                except Exception as exc:
                    item.update(status='error', error=str(exc))
                if item['status'] == 'ok' and item['missing_ranks']:
                    item['status'] = 'partial_missing_ranks'
                completed.add(signature)
                atomic_json(output / 'manifest.json', manifest)
                print('CHECKPOINT_END', item['checkpoint'], item['status'], flush=True)
    summarize(output, manifest)
    print('REPORT', output / 'report.md', flush=True)


if __name__ == '__main__':
    main()
