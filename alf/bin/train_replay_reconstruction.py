"""Prepare or run an isolated SAC/RLPD historical-replay continuation."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from alf.utils.replay_reconstruction import MIXTURE, _digest, _json, _load


def _preconfig(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'pre_config'):
            return ast.literal_eval(node.args[0])
    raise ValueError(f'Missing literal pre_config: {path}')


def inspect_source(source):
    source = Path(source).resolve()
    hints = _preconfig(source / 'alf_config.py')
    if hints.get('create_environment.env_name') != 'dog:walk':
        raise ValueError('Expected a dog:walk source run')
    config_files = list((source / 'config_files').rglob('*.py'))
    kinds = [kind for kind in ('sac', 'rlpd')
             if (source / 'config_files' / f'{kind}_dmc_conf.py').is_file()]
    if len(kinds) != 1 or hints.get('TrainerConfig.num_env_steps') != 600000:
        raise ValueError('Expected an archived 600k SAC or RLPD job')
    paths = [source / 'alf_config.py', *config_files]
    for checkpoint, _ in MIXTURE:
        p = source / 'train/algorithm' / f'ckpt-{checkpoint}'
        model = _load(p)
        if (int(model['trainer_progress']['_env_steps']) != checkpoint * 1000 // 1001
                or int(model['global_step']) != checkpoint):
            raise ValueError(f'Unexpected checkpoint horizon: {p}')
        paths.append(p)
    optimizer = source / 'train/algorithm/ckpt-150150-optimizer'
    _load(optimizer)
    paths.append(optimizer)
    # Shared imported config must still match the archived job, not a new default.
    relative = 'alf/examples/benchmarks/dm_control/dmc_conf.py'
    shared = Path(__file__).resolve().parents[2] / relative
    with tarfile.open(source / 'alf.tar.gz') as archive:
        if archive.extractfile(relative).read() != shared.read_bytes():
            raise ValueError('Shared DMC config differs from the archived source')
    paths.extend([shared, source / 'alf.tar.gz'])
    return dict(source_run=str(source), algorithm=kinds[0],
                seed=int(hints['TrainerConfig.random_seed']),
                inputs={str(p): _digest(p) for p in paths})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True)
    p.add_argument('--root-dir', required=True)
    p.add_argument('--worker-gpus', default='0,1,2,3')
    p.add_argument('--port', type=int, default=0, help='HTTP port; 0 chooses a free port')
    p.add_argument('--final-env-steps', type=int, default=800000,
                   help='Absolute total horizon across four workers')
    p.add_argument('--reconstruction-seed', type=int, default=20260921)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--reconstruct-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def prepare(args):
    root = Path(args.root_dir).resolve()
    source = Path(args.source_run).resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('Destination must be separate from source')
    devices = args.worker_gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or not all(d.isdigit() for d in devices):
        raise ValueError('Expected four distinct GPU indices')
    if args.final_env_steps <= 600000 or args.final_env_steps % 4:
        raise ValueError('Final total horizon must exceed 600k and be divisible by four')
    options = dict(inspect_source(source), root_dir=str(root), version=1,
                   final_env_steps=args.final_env_steps,
                   reconstruction_seed=args.reconstruction_seed,
                   mixture=[list(x) for x in MIXTURE])
    manifest = root / 'reconstruction_manifest.json'
    if manifest.exists():
        if not args.resume:
            raise ValueError('Destination already prepared; use --resume')
        saved = json.loads(manifest.read_text())
        if any(saved.get(k) != v for k, v in options.items()):
            raise ValueError('Source or reconstruction settings changed')
        for relative, digest in saved['config_hashes'].items():
            if _digest(root / relative) != digest:
                raise ValueError(f'Saved configuration changed: {relative}')
        return saved
    if args.resume:
        raise ValueError('--resume requires an existing reconstruction manifest')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Destination must be empty')
    if args.dry_run:
        return options
    root.mkdir(parents=True, exist_ok=True)
    confdir = root / 'source_config'
    confdir.mkdir()
    shutil.copy2(source / 'alf_config.py', confdir / 'alf_config.py')
    shutil.copytree(source / 'config_files', confdir / 'config_files')
    config = ('import json, os\nfrom pathlib import Path\nimport alf\n'
              'from alf.utils.replay_reconstruction import configure\n'
              f'alf.import_config({str(confdir / "alf_config.py")!r})\n'
              f'options = json.loads(Path({str(manifest)!r}).read_text())\n'
              'options["reconstruct_only"] = os.environ.get("ALF_REPLAY_RECONSTRUCT_ONLY") == "1"\n'
              'configure(options)\n')
    (root / 'reconstruction_conf.py').write_text(config)
    options['config_hashes'] = {
        str(p.relative_to(root)): _digest(p) for p in
        [root / 'reconstruction_conf.py', *confdir.rglob('*.py')]}
    _json(manifest, options)
    return options


def main():
    args = parser().parse_args()
    options = prepare(args)
    command = [sys.executable, '-m', 'alf.bin.train',
               '--root_dir', options['root_dir'],
               '--conf', str(Path(options['root_dir']) / 'reconstruction_conf.py'),
               '--distributed=multi-gpu', '--worker_gpus', args.worker_gpus,
               '--port', str(args.port), '--alsologtostderr']
    if args.dry_run:
        import shlex
        print(json.dumps(options, indent=2))
        prefix = "ALF_REPLAY_RECONSTRUCT_ONLY=1 " if args.reconstruct_only else ""
        print(prefix + shlex.join(command))
    elif not args.prepare_only:
        env = dict(os.environ, ALF_REPLAY_RECONSTRUCT_ONLY='1' if args.reconstruct_only else '0')
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
