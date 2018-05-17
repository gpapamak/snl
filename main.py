import argparse
import numpy as np

import util.io

import experiment_descriptor as ed
import misc


def parse_args():
    """
    Returns an object describing the command line.
    """

    parser = argparse.ArgumentParser(description='Likelihood-free inference experiments.')
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser('run', help='run experiments')
    parser_run.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_run.set_defaults(func=run_experiment)

    parser_trials = subparsers.add_parser('trials', help='run multiple experiment trials')
    parser_trials.add_argument('start', type=int, help='# of first trial')
    parser_trials.add_argument('end', type=int, help='# of last trial')
    parser_trials.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_trials.set_defaults(func=run_trials)

    parser_view = subparsers.add_parser('view', help='view results')
    parser_view.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_view.add_argument('-b', '--block', action='store_true', help='block execution after viewing each experiment')
    parser_view.add_argument('-t', '--trial', type=int, default=0, help='trial to view (default is 0)')
    parser_view.set_defaults(func=view_results)

    parser_log = subparsers.add_parser('log', help='print experiment logs')
    parser_log.add_argument('files', nargs='+', type=str, help='file(s) describing experiments')
    parser_log.set_defaults(func=print_log)

    return parser.parse_args()


def run_experiment(args):
    """
    Runs experiments.
    """

    from experiment_runner import ExperimentRunner

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        try:
            ExperimentRunner(exp_desc).run(trial=0, sample_gt=False, rng=np.random.RandomState(42))

        except misc.AlreadyExistingExperiment:
            print 'EXPERIMENT ALREADY EXISTS'

    print 'ALL DONE'


def run_trials(args):
    """
    Runs experiments for multiple trials with random ground truth.
    """

    from experiment_runner import ExperimentRunner

    if args.start < 1:
        raise ValueError('trial # must be a positive integer')

    if args.end < args.start:
        raise ValueError('end trial can''t be less than start trial')

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        runner = ExperimentRunner(exp_desc)

        for trial in xrange(args.start, args.end + 1):

            try:
                runner.run(trial=trial, sample_gt=True, rng=np.random)

            except misc.AlreadyExistingExperiment:
                print 'EXPERIMENT ALREADY EXISTS'

    print 'ALL DONE'


def view_results(args):
    """
    Views experiments.
    """

    from experiment_viewer import ExperimentViewer, plt

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        try:
            ExperimentViewer(exp_desc).view_results(trial=args.trial, block=args.block)

        except misc.NonExistentExperiment:
            print 'EXPERIMENT DOES NOT EXIST'

    plt.show()


def print_log(args):
    """
    Prints experiment logs.
    """

    from experiment_viewer import ExperimentViewer

    exp_descs = sum([ed.parse(util.io.load_txt(f)) for f in args.files], [])

    for exp_desc in exp_descs:

        try:
            ExperimentViewer(exp_desc).print_log()

        except misc.NonExistentExperiment:
            print 'EXPERIMENT DOES NOT EXIST'


def main():

    args = parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
