import os
import re
import util.misc


class ParseError(Exception):
    """
    Exception to be thrown when there is a parsing error.
    """

    def __init__(self, str):
        self.str = str

    def __str__(self):
        return self.str


class InferenceDescriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('rej_abc', str):
            return Rej_ABC_Descriptor(str)

        elif re.match('mcmc_abc', str):
            return MCMC_ABC_Descriptor(str)

        elif re.match('smc_abc', str):
            return SMC_ABC_Descriptor(str)

        elif re.match('synth_lik', str):
            return SynthLik_Descriptor(str)

        elif re.match('nde', str):
            return NDE_Descriptor(str)

        elif re.match('post_prop', str):
            return PostProp_Descriptor(str)

        elif re.match('snpe_mdn', str):
            return SNPE_MDN_Descriptor(str)

        elif re.match('snl', str):
            return SNL_Descriptor(str)

        else:
            raise ParseError(str)


class ABC_Descriptor(InferenceDescriptor):

    def get_id(self):

        raise NotImplementedError('abstract method')

    def get_dir(self):

        return os.path.join('abc', self.get_id())


class Rej_ABC_Descriptor(ABC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.eps = None
        self.parse(str)

    def pprint(self):

        str = 'rej_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\teps: {0}\n'.format(self.eps)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'rej_abc\{n_samples:(.*),eps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.eps = float(m.group(2))

    def get_id(self, delim='_'):

        id = 'rejabc'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'eps' + delim + str(self.eps)

        return id


class MCMC_ABC_Descriptor(ABC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.eps = None
        self.step = None
        self.parse(str)

    def pprint(self):

        str = 'mcmc_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\teps: {0},\n'.format(self.eps)
        str += '\t\tstep: {0}\n'.format(self.step)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'mcmc_abc\{n_samples:(.*),eps:(.*),step:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.eps = float(m.group(2))
        self.step = float(m.group(3))

    def get_id(self, delim='_'):

        id = 'mcmcabc'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'eps' + delim + str(self.eps)
        id += delim + 'step' + delim + str(self.step)

        return id


class SMC_ABC_Descriptor(ABC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.eps_init = None
        self.eps_last = None
        self.eps_decay = None
        self.parse(str)

    def pprint(self):

        str = 'smc_abc\n'
        str += '\t{\n'
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\teps_init: {0},\n'.format(self.eps_init)
        str += '\t\teps_last: {0},\n'.format(self.eps_last)
        str += '\t\teps_decay: {0}\n'.format(self.eps_decay)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'smc_abc\{n_samples:(.*),eps_init:(.*),eps_last:(.*),eps_decay:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.eps_init = float(m.group(2))
        self.eps_last = float(m.group(3))
        self.eps_decay = float(m.group(4))

    def get_id(self, delim='_'):

        id = 'smcabc'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'epsinit' + delim + str(self.eps_init)
        id += delim + 'epslast' + delim + str(self.eps_last)
        id += delim + 'epsdecay' + delim + str(self.eps_decay)

        return id


class MCMC_Descriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('gauss_metropolis', str):
            return GaussianMetropolisDescriptor(str)

        elif re.match('slice_sampler', str):
            return SliceSamplerDescriptor(str)

        else:
            raise ParseError(str)


class GaussianMetropolisDescriptor(MCMC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.step = None
        self.parse(str)

    def pprint(self):

        str = 'gauss_metropolis\n'
        str += '\t\t{\n'
        str += '\t\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\t\tstep: {0}\n'.format(self.step)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'gauss_metropolis\{n_samples:(.*),step:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))
        self.step = float(m.group(2))

    def get_id(self, delim='_'):

        id = 'gaussmetropolis'
        id += delim + 'samples' + delim + str(self.n_samples)
        id += delim + 'step' + delim + str(self.step)

        return id


class SliceSamplerDescriptor(MCMC_Descriptor):

    def __init__(self, str):

        self.n_samples = None
        self.parse(str)

    def pprint(self):

        str = 'slice_sampler\n'
        str += '\t\t{\n'
        str += '\t\t\tn_samples: {0}\n'.format(self.n_samples)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'slice_sampler\{n_samples:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_samples = int(m.group(1))

    def get_id(self, delim='_'):

        id = 'slicesampler'
        id += delim + 'samples' + delim + str(self.n_samples)

        return id


class SynthLik_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.mcmc = None
        self.n_sims = None
        self.parse(str)

    def pprint(self):

        str = 'synth_lik\n'
        str += '\t{\n'
        str += '\t\tmcmc: {0},\n'.format(self.mcmc.pprint())
        str += '\t\tn_sims: {0}\n'.format(self.n_sims)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'synth_lik\{mcmc:(.*),n_sims:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.mcmc = MCMC_Descriptor.get_descriptor(m.group(1))
        self.n_sims = int(m.group(2))

    def get_dir(self):

        return os.path.join('synthlik_sims_{0}'.format(self.n_sims), self.mcmc.get_id())


class ModelDescriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('mdn', str):
            return MDN_Descriptor(str)

        elif re.match('maf', str):
            return MAF_Descriptor(str)

        else:
            raise ParseError(str)


class MDN_Descriptor(ModelDescriptor):

    def __init__(self, str):

        self.n_hiddens = None
        self.act_fun = None
        self.n_comps = None
        self.parse(str)

    def pprint(self):

        str = 'mdn\n'
        str += '\t\t{\n'
        str += '\t\t\tn_hiddens: {0},\n'.format(self.n_hiddens)
        str += '\t\t\tact_fun: {0},\n'.format(self.act_fun)
        str += '\t\t\tn_comps: {0}\n'.format(self.n_comps)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'mdn\{n_hiddens:\[(.*)\],act_fun:(.*),n_comps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_hiddens = map(int, m.group(1).split(',')) if m.group(1) else []
        self.act_fun = m.group(2)
        self.n_comps = int(m.group(3))

    def get_id(self, delim='_'):

        id = 'mdn' + delim + 'hiddens' + delim

        for h in self.n_hiddens:
            id += str(h) + delim

        id += 'comps' + delim + str(self.n_comps) + delim
        id += self.act_fun

        return id


class MAF_Descriptor(ModelDescriptor):

    def __init__(self, str):

        self.n_hiddens = None
        self.act_fun = None
        self.n_comps = None
        self.parse(str)

    def pprint(self):

        str = 'maf\n'
        str += '\t\t{\n'
        str += '\t\t\tn_hiddens: {0},\n'.format(self.n_hiddens)
        str += '\t\t\tact_fun: {0},\n'.format(self.act_fun)
        str += '\t\t\tn_comps: {0}\n'.format(self.n_comps)
        str += '\t\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'maf\{n_hiddens:\[(.*)\],act_fun:(.*),n_comps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_hiddens = map(int, m.group(1).split(',')) if m.group(1) else []
        self.act_fun = m.group(2)
        self.n_comps = int(m.group(3))

    def get_id(self, delim='_'):

        id = 'maf' + delim + 'hiddens' + delim

        for h in self.n_hiddens:
            id += str(h) + delim

        id += 'comps' + delim + str(self.n_comps) + delim
        id += self.act_fun

        return id


class NDE_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.target = None
        self.n_samples = None
        self.parse(str)

    def pprint(self):

        str = 'nde\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\ttarget: {0},\n'.format(self.target)
        str += '\t\tn_samples: {0}\n'.format(self.n_samples)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'nde\{model:(.*),target:(posterior|likelihood),n_samples:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = ModelDescriptor.get_descriptor(m.group(1))
        self.target = m.group(2)
        self.n_samples = int(m.group(3))

    def get_dir(self):

        return os.path.join('nde', '{0}_samples_{1}'.format(self.target, self.n_samples), self.model.get_id())


class PostProp_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.n_samples_p = None
        self.n_rounds_p = None
        self.maxepochs_p = None
        self.n_samples_f = None
        self.maxepochs_f = None
        self.parse(str)

    def pprint(self):

        str = 'post_prop\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\tn_samples_p: {0},\n'.format(self.n_samples_p)
        str += '\t\tn_rounds_p: {0},\n'.format(self.n_rounds_p)
        str += '\t\tmaxepochs_p: {0},\n'.format(self.maxepochs_p)
        str += '\t\tn_samples_f: {0},\n'.format(self.n_samples_f)
        str += '\t\tmaxepochs_f: {0}\n'.format(self.maxepochs_f)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'post_prop\{model:(.*),n_samples_p:(.*),n_rounds_p:(.*),maxepochs_p:(.*),n_samples_f:(.*),maxepochs_f:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = MDN_Descriptor(m.group(1))
        self.n_samples_p = int(m.group(2))
        self.n_rounds_p = int(m.group(3))
        self.maxepochs_p = int(m.group(4))
        self.n_samples_f = int(m.group(5))
        self.maxepochs_f = int(m.group(6))

    def get_dir(self):

        return os.path.join('postprop_samplesp_{0}_roundsp_{1}_maxepochsp_{2}_samplesf_{3}_maxepochsf_{4}'.format(
            self.n_samples_p, self.n_rounds_p, self.maxepochs_p, self.n_samples_f, self.maxepochs_f),
            self.model.get_id()
        )


class SNPE_MDN_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.maxepochs = None
        self.parse(str)

    def pprint(self):

        str = 'snpe_mdn\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\tn_rounds: {0},\n'.format(self.n_rounds)
        str += '\t\tmaxepochs: {0}\n'.format(self.maxepochs)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'snpe_mdn\{model:(.*),n_samples:(.*),n_rounds:(.*),maxepochs:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = MDN_Descriptor(m.group(1))
        self.n_samples = int(m.group(2))
        self.n_rounds = int(m.group(3))
        self.maxepochs = int(m.group(4))

    def get_dir(self):

        return os.path.join('snpemdn_samples_{0}_rounds_{1}_maxepochs_{2}'.format(
            self.n_samples, self.n_rounds, self.maxepochs),
            self.model.get_id()
        )


class SNL_Descriptor(InferenceDescriptor):

    def __init__(self, str):

        self.model = None
        self.n_samples = None
        self.n_rounds = None
        self.train_on = None
        self.thin = None
        self.parse(str)

    def pprint(self):

        str = 'snl\n'
        str += '\t{\n'
        str += '\t\tmodel: {0},\n'.format(self.model.pprint())
        str += '\t\tn_samples: {0},\n'.format(self.n_samples)
        str += '\t\tn_rounds: {0},\n'.format(self.n_rounds)
        str += '\t\ttrain_on: {0},\n'.format(self.train_on)
        str += '\t\tthin: {0}\n'.format(self.thin)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'snl\{model:(.*),n_samples:(.*),n_rounds:(.*),train_on:(all|last),thin:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.model = ModelDescriptor.get_descriptor(m.group(1))
        self.n_samples = int(m.group(2))
        self.n_rounds = int(m.group(3))
        self.train_on = m.group(4)
        self.thin = int(m.group(5))

    def get_dir(self):

        return os.path.join('snl_samples_{0}_rounds_{1}_train_on_{2}_thin_{3}'.format(self.n_samples, self.n_rounds, self.train_on, self.thin), self.model.get_id())


class ExperimentDescriptor:

    def __init__(self, str):

        self.sim = None
        self.inf = None
        self.parse(str)

    def pprint(self):

        str = 'experiment\n'
        str += '{\n'
        str += '\tsim: {0},\n'.format(self.sim)
        str += '\tinf: {0}\n'.format(self.inf.pprint())
        str += '}\n'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'experiment\{sim:(mg1|lotka_volterra|gauss|hodgkin_huxley),inf:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.sim = m.group(1)
        self.inf = InferenceDescriptor.get_descriptor(m.group(2))

    def get_dir(self):

        return os.path.join(self.sim, self.inf.get_dir())


def parse(str):
    """
    Parses the string str, and returns a list of experiment descriptor objects described by the string.
    """

    str = util.misc.remove_whitespace(str)
    descs = []

    pattern = re.compile(r'experiment\{')
    match = pattern.search(str)

    while match:

        exp_str = match.group()
        left = 1
        right = 0
        i = match.end()

        # consume the string until curly brackets close
        while left > right:

            try:
                if str[i] == '{':
                    left += 1

                if str[i] == '}':
                    right += 1

            # if we have reached the end of the string, discard current match
            except IndexError:
                print 'Experiment not compiled. End reached without bracket closing.'
                print ''
                return descs

            exp_str += str[i]
            i += 1

        try:
            desc = ExperimentDescriptor(exp_str)
            descs.append(desc)

        except ParseError as err:
            print 'Experiment not compiled. Parse error in:'
            print err
            print ''

        str = str[i:]
        match = pattern.search(str)

    return descs
