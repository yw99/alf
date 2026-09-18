"""Start or validate a faithful BAFCv3 -> TR2 continuation.

See docs/bafcv3_tr2_restart.md. Full training uses alf.bin.train's existing
four-GPU launcher; --validate-only uses four isolated Gloo workers and writes a
calibrated initial checkpoint without any environment or optimizer steps.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch

from alf.bin.evaluate_bafcv3_checkpoints import atomic_json, digest, jsonable, preconfig
from alf.utils.bafcv3_restart import WORLD_SIZE, VERSION, fingerprint_inputs, validate_options, settings_fingerprint, verify_config_snapshot


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-checkpoint', required=True)
    p.add_argument('--root-dir', required=True)
    p.add_argument('--critic-utd', type=int, default=3)
    p.add_argument('--threshold-quantile', type=float, default=.33)
    p.add_argument('--calibration-repetitions', type=int, default=100)
    p.add_argument('--calibration-seed', type=int, default=20260917)
    p.add_argument('--rollout-skipping', choices=('on', 'off'), default='on')
    p.add_argument('--final-env-steps-per-rank', type=int, default=150000)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker-gpus', default='0,1,2,3')
    p.add_argument('--validation-device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--smoke-train-iters', type=int, default=0,
                   help='Validation only: bounded replay training iterations after saving the initial checkpoint (0..2).')
    p.add_argument('--smoke-env-steps', type=int, default=0,
                   help='Validation only: collect 0..2 real DM Control steps after the initial checkpoint.')
    p.add_argument('--prepare-only', action='store_true', help='Write configuration and provenance, then exit.')
    return p


def prepare(args):
    root = Path(args.root_dir).resolve()
    source = str(Path(args.source_checkpoint).resolve())
    source_run = Path(source).parents[2]
    if root.is_relative_to(source_run) or source_run.is_relative_to(root):
        raise ValueError('Output must be separate from the source run')
    settings = dict(source_checkpoint=source, critic_utd=args.critic_utd,
                    threshold_quantile=args.threshold_quantile,
                    calibration_repetitions=args.calibration_repetitions,
                    calibration_seed=args.calibration_seed,
                    rollout_skipping=args.rollout_skipping == 'on',
                    final_env_steps_per_rank=args.final_env_steps_per_rank)
    validate_options(settings)
    inputs = fingerprint_inputs(source)
    key = settings_fingerprint(settings, inputs)
    manifest_path = root / 'restart_manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise ValueError('Output already exists; use --resume with identical experiment settings')
        options = json.loads(manifest_path.read_text())
        if (options['settings_fingerprint'] != key or
                any(options.get(k) != v for k, v in settings.items())):
            raise ValueError('Inputs or experiment settings changed; use a new output directory')
        verify_config_snapshot(options)
        return options
    if args.resume:
        raise ValueError('--resume requires an existing restart manifest')
    if root.exists() and any(root.iterdir()):
        raise ValueError('New restart output directory must be empty')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    env_steps = int(checkpoint['trainer_progress']['_env_steps'])
    if env_steps >= args.final_env_steps_per_rank:
        raise ValueError('Final horizon must be after the checkpoint environment step')
    root.mkdir(parents=True, exist_ok=True)
    # Snapshot saved configuration so restarts do not depend on mutable copies.
    conf_dir = root / 'source_config'
    conf_dir.mkdir()
    run = Path(source).parents[2]
    shutil.copy2(run / 'alf_config.py', conf_dir / 'alf_config.py')
    shutil.copytree(run / 'config_files', conf_dir / 'config_files')
    repo = Path(__file__).resolve().parents[2]
    code_files = ['alf/utils/bafcv3_restart.py', 'alf/bin/train_bafcv3_tr2_restart.py',
                  'alf/algorithms/bafc_algorithm_v3.py', 'alf/algorithms/bafc_algorithm_v3_tr2.py',
                  'alf/algorithms/agent.py', 'alf/trainers/policy_trainer.py',
                  'alf/experience_replayers/replay_buffer.py', 'alf/utils/checkpoint_utils.py']
    hints = preconfig(run / 'alf_config.py')
    for name in code_files:
        destination = root / 'code_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, destination)
    options = dict(**settings, root_dir=str(root), inputs=inputs, version=VERSION,
                   task=hints.get('create_environment.env_name'), seed=hints.get('TrainerConfig.random_seed'),
                   settings_fingerprint=key, source_env_steps=env_steps,
                   created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   code_hashes={p: digest(repo / p) for p in code_files},
                   dependencies={'python': sys.version, 'torch': torch.__version__,
                                 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8')},
                   gpu_available=torch.cuda.is_available(),
                   source_global_step=int(checkpoint['global_step']))
    conf = ('import json\nfrom pathlib import Path\nimport alf\n'
            'from alf.utils.bafcv3_restart import configure\n'
            f'alf.import_config({str(conf_dir / "alf_config.py")!r})\n'
            f'configure(json.loads(Path({str(manifest_path)!r}).read_text()))\n')
    (root / 'restart_conf.py').write_text(conf)
    options['config_hashes'] = {str(path.relative_to(root)): digest(path)
        for path in [root / 'restart_conf.py', *sorted(conf_dir.rglob('*.py'))]}
    atomic_json(manifest_path, options)
    return options


class _ValidationEnvironment:
    """Specifications and FIRST only; never fabricates training transitions."""
    batch_size = 1

    def __init__(self, observation_spec, action_spec):
        self.obs, self.action = observation_spec, action_spec

    def reset(self):
        from alf.data_structures import TimeStep, StepType
        return TimeStep(step_type=torch.full((1,), StepType.FIRST, dtype=torch.int32),
                        reward=torch.zeros(1), discount=torch.ones(1),
                        observation=self.obs.zeros((1,)), prev_action=self.action.zeros((1,)),
                        env_id=torch.zeros(1, dtype=torch.int32),
                        env_info={'num_env_frames': torch.zeros(1, dtype=torch.int64)})


def validation_worker(rank, options, rendezvous, device, smoke_iters, gpu_indices, smoke_env_steps):
    import alf
    from alf.algorithms.agent import Agent
    from alf.algorithms.config import TrainerConfig
    from alf.algorithms.data_transformer import create_data_transformer
    from alf.tensor_specs import TensorSpec, BoundedTensorSpec
    from alf.trainers.policy_trainer import TrainerProgress
    from alf.utils.common import parse_conf_file
    from alf.utils.per_process_context import PerProcessContext
    from alf.utils.checkpoint_utils import Checkpointer
    from alf.utils.bafcv3_restart import restore, state_digest, atomic_json, collective_call, clone_empty_optimizer, compare_cpu_gpu_metric
    from alf.config_helpers import adjust_config_by_multi_process_divider
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.cuda.set_device(gpu_indices[rank])
        alf.set_default_device("cuda")
    torch.distributed.init_process_group('gloo', init_method=rendezvous,
        rank=rank, world_size=WORLD_SIZE, timeout=datetime.timedelta(minutes=30))
    try:
        PerProcessContext().set_distributed(rank=rank, local_rank=rank, num_processes=WORLD_SIZE)
        PerProcessContext().finalize()
        parse_conf_file(str(Path(options['root_dir']) / 'restart_conf.py'), create_env=False)
        adjust_config_by_multi_process_divider(rank, WORLD_SIZE)
        model = torch.load(options['source_checkpoint'], map_location='cpu', weights_only=True)
        obs = TensorSpec((model['algorithm']['_rl_algorithm._actor_eval_samples'].shape[-1],))
        action = BoundedTensorSpec((model['algorithm']['_rl_algorithm._actor_networks._action_layer._bias'].shape[-1],), minimum=-1., maximum=1.)
        pristine_optimizer = clone_empty_optimizer(alf.get_config_value('Agent.optimizer'))
        def construct(ckpt_dir=None):
            config = TrainerConfig(root_dir=options['root_dir'])
            config.data_transformer = create_data_transformer(config.data_transformer_ctor, obs)
            agent = Agent(observation_spec=obs, action_spec=action, config=config,
                          env=_ValidationEnvironment(obs, action),
                          optimizer=clone_empty_optimizer(pristine_optimizer), debug_summaries=True)
            agent.set_path('')
            agent.activate_ddp(rank)
            progress = TrainerProgress()
            progress.set_termination_criterion(config.num_iterations, config.num_env_steps)
            metrics = torch.nn.ModuleList(agent.get_metrics())
            cp = Checkpointer(ckpt_dir=str(ckpt_dir or Path(options['root_dir']) / 'train/algorithm'),
                              algorithm=agent, metrics=metrics, trainer_progress=progress)
            return agent, progress, metrics, cp
        agent, progress, metrics, cp = collective_call(construct)
        restore(agent, progress, metrics, cp, options, rank)
        # Verify a fresh construction resumes the checkpoint exactly and does not
        # invoke train_iter or recalibrate. Capture state before any smoke updates.
        before = state_digest(agent.state_dict())
        threshold = agent._rl_algorithm._eval_trust_max
        del agent, progress, metrics, cp
        agent, progress, metrics, cp = collective_call(construct)
        restore(agent, progress, metrics, cp, options, rank)
        after = state_digest(agent.state_dict())
        if before != after or threshold != agent._rl_algorithm._eval_trust_max:
            raise ValueError('Native resume changed calibrated training state')
        comparison = collective_call(lambda: compare_cpu_gpu_metric(agent)) if device == 'cuda' else None
        result = dict(rank=rank, device=device, cpu_gpu_metric_comparison=comparison, restored=True, calibration_unchanged=True,
                      exact_native_resume=True, initial_optimizer_steps=0,
                      initial_environment_steps=0, threshold=threshold,
                      progress=jsonable(progress.state_dict()), smoke_train_iters=smoke_iters)
        if smoke_env_steps:
            from alf.environments.utils import create_environment
            from alf.utils.common import set_global_env
            env = collective_call(lambda: create_environment(nonparallel=True,
                num_parallel_environments=1, seed=int(options['seed']) + rank))
            try:
                agent._env = env
                set_global_env(env)
                env.reset()
                agent.reset_state()
                positions = agent._replay_buffer._current_pos.clone()
                # Exercise genuine rollout and replay insertion without forcing
                # a fake transition or depending on whether the gate would skip.
                collective_call(lambda: agent.unroll(smoke_env_steps))
                if not torch.equal(agent._replay_buffer._current_pos, positions + smoke_env_steps):
                    raise ValueError('Real rollout did not append the expected replay entries')
                result['real_environment_rollout_steps'] = smoke_env_steps
            finally:
                env.close()
                agent.reset_state()
        if smoke_iters:
            alg = agent._rl_algorithm
            counts = (alg._critic_update_counter, alg._actor_update_counter)
            # The regular learner path, using real saved replay. No environment
            # step is fabricated by the spec-only validation environment.
            agent.activate_ddp(rank)
            for _ in range(smoke_iters):
                agent.train_from_replay_buffer(update_global_counter=False)
            deltas = [alg._critic_update_counter - counts[0], alg._actor_update_counter - counts[1]]
            expected = [smoke_iters * 12 * options['critic_utd'] // (options['critic_utd'] + 1),
                        smoke_iters * 12 // (options['critic_utd'] + 1)]
            if deltas != expected:
                raise ValueError(f'Wrong update schedule: {deltas} != {expected}')
            for name, network in [('actor', alg._actor_networks), ('critic', alg._critic_networks)]:
                hashes = [None] * WORLD_SIZE
                torch.distributed.all_gather_object(hashes, state_digest(network.state_dict()))
                if len(set(hashes)) != 1:
                    raise ValueError(f'{name} parameters diverged across DDP ranks')
                result[name + '_parameters_synchronized'] = True
            result['smoke_update_counts_critic_actor'] = deltas
            # Retain the pristine initial checkpoint. Save the bounded smoke
            # state separately and verify rank-local state and learner resume.
            smoke_dir = Path(options['root_dir']) / 'validation_smoke/algorithm'
            smoke_cp = Checkpointer(ckpt_dir=str(smoke_dir), algorithm=agent,
                                   metrics=metrics, trainer_progress=progress)
            collective_call(lambda: smoke_cp.save(options['source_global_step'] + 1,
                                                   ddp_rank=rank))
            def learner_digest(current):
                return state_digest({k: v for k, v in current.state_dict().items()
                                     if k != '_replay_buffer._restart_boundaries'})
            expected_state = learner_digest(agent)
            expected_local = state_digest(agent._rank_local_checkpoint_state())
            del agent, alg, progress, metrics, cp, smoke_cp
            agent, progress, metrics, cp = collective_call(lambda: construct(smoke_dir))
            restore(agent, progress, metrics, cp, options, rank)
            if learner_digest(agent) != expected_state:
                raise ValueError('Post-learning checkpoint changed learner state')
            if state_digest(agent._rank_local_checkpoint_state()) != expected_local:
                raise ValueError('Post-learning checkpoint changed rank-local state or RNG')
            result['exact_post_learning_resume'] = True
        collective_call(lambda: atomic_json(Path(options['root_dir']) / f'validation_rank{rank}.json', result))
    finally:
        torch.distributed.destroy_process_group()


def mapped_validation_worker(rank, world_size, options, rendezvous, smoke_iters, smoke_env_steps):
    """Use the same one-visible-GPU-per-rank launcher as full training."""
    if world_size != WORLD_SIZE or torch.cuda.device_count() != 1:
        raise ValueError('Expected four workers with one visible GPU each')
    validation_worker(rank, options, rendezvous, 'cuda', smoke_iters,
                      [0] * WORLD_SIZE, smoke_env_steps)


def main(argv=None):
    # Must precede CUDA initialization in every fresh training worker.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parser().parse_args(argv)
    if not 0 <= args.smoke_train_iters <= 2 or (args.smoke_train_iters and not args.validate_only):
        raise ValueError('--smoke-train-iters requires --validate-only and a value in 0..2')
    gpu_indices = [int(i) for i in args.worker_gpus.split(',')]
    if not 0 <= args.smoke_env_steps <= 2 or (args.smoke_env_steps and not args.validate_only):
        raise ValueError('--smoke-env-steps requires --validate-only and a value in 0..2')
    if len(gpu_indices) != WORLD_SIZE or len(set(gpu_indices)) != WORLD_SIZE or min(gpu_indices) < 0:
        raise ValueError('Exactly four training workers are required')
    options = prepare(args)
    if args.prepare_only:
        print(f'Prepared {options["root_dir"]}')
        return
    if args.validate_only:
        if args.validation_device == 'cuda' and torch.cuda.device_count() <= max(gpu_indices):
            raise RuntimeError('Four usable CUDA devices are required for GPU validation')
        with tempfile.TemporaryDirectory(prefix='bafcv3-restart-') as temp:
            rendezvous = 'file://' + temp + '/rendezvous'
            if args.validation_device == 'cuda':
                from alf.utils.worker_gpu import resolve_worker_gpus, run_workers
                devices = resolve_worker_gpus(args.worker_gpus,
                    os.environ.get('CUDA_VISIBLE_DEVICES'), torch.cuda.device_count())
                run_workers(devices, mapped_validation_worker,
                    (options, rendezvous, args.smoke_train_iters, args.smoke_env_steps))
            else:
                torch.multiprocessing.spawn(validation_worker,
                    args=(options, rendezvous, 'cpu', args.smoke_train_iters,
                          gpu_indices, args.smoke_env_steps),
                    nprocs=WORLD_SIZE, join=True)
        current = fingerprint_inputs(options['source_checkpoint'])
        if current != options['inputs']:
            raise RuntimeError('Source input fingerprints changed during validation')
        atomic_json(Path(options['root_dir']) / 'validation.json',
                    {'status': 'passed', 'source_files_unchanged': True,
                     'device': args.validation_device, 'world_size': WORLD_SIZE,
                     'smoke_train_iters': args.smoke_train_iters,
                     'real_environment_rollout_steps': args.smoke_env_steps,
                     'gpu_validation': 'passed' if args.validation_device == 'cuda' else 'not_run'})
        print(f'Validated {options["root_dir"]}')
        return
    if torch.cuda.device_count() <= max(gpu_indices):
        raise RuntimeError('Full training requires four usable GPUs; configuration is prepared')
    # Calling .venv/bin/python directly does not activate its companion tools
    # (notably Ninja, used by ALF's native environment extension).
    child_env = dict(os.environ)
    child_env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + child_env.get('PATH', '')
    # prepare() already snapshots the source configuration and restart code.
    # ALF's generic repository snapshot forbids our in-repository artifact roots.
    # Let the OS assign each job's optional HTTP control port to avoid collisions.
    subprocess.run([sys.executable, '-m', 'alf.bin.train',
                    '--conf', str(Path(options['root_dir']) / 'restart_conf.py'),
                    '--root_dir', options['root_dir'], '--distributed=multi-gpu',
                    '--worker_gpus', args.worker_gpus, '--nostore_snapshot', '--port=0',
                    '--alsologtostderr'], check=True, env=child_env)


if __name__ == '__main__':
    main()
