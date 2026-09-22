"""Faithful BAFCv3 -> TR2 continuation with automatic rollout-gate activation."""
import argparse
from alf.bin import train_bafcv3_tr2_restart as restart
from alf.utils.bafcv3_auto_skip import DEFAULTS


class AutoSkipParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args, namespace)
        args.auto_skip = {key: getattr(args, 'auto_' + key) for key in DEFAULTS}
        if args.rollout_skipping != 'off':
            self.error('Automatic activation requires --rollout-skipping off initially')
        return args


def parser():
    p = AutoSkipParser(description=__doc__, parents=[restart.parser()],
                      add_help=False)
    p.set_defaults(critic_utd=11, rollout_skipping='off',
                   final_env_steps_per_rank=200000)
    for key, value in DEFAULTS.items():
        p.add_argument('--auto-' + key.replace('_', '-'),
                       type=type(value), default=value)
    return p


def main(argv=None):
    restart.main(argv, argument_parser=parser())


if __name__ == '__main__':
    main()
