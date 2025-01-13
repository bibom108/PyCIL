import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import json
import argparse
from utils import factory
from utils.data_manager import DataManager
from time import time
import datetime
from utils.toolkit import count_parameters
from utils.inc_net import DERNet
import copy
from torch.utils.data import DataLoader


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):

                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                outputs = self.base_classifier(batch + noise)['logits']
                predictions = torch.max(outputs, dim=1)[1]

                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


batch = 1
skip = 1
max = -1
N0 = 100
N  = 1000
alpha = 0.001
sigma = 0.25
classes=[10, 20]
ww = 0
weight="weights/_3.pkl"
outfile = "certify/all.txt"

f = open(outfile, 'w')
print("", file=f, flush=True)

param = load_json("./exps/der.json")
args = argparse.ArgumentParser()
args = vars(args)
args.update(param)

seed_list = copy.deepcopy(args['seed'])
device = copy.deepcopy(args['device'])

for seed in seed_list:
    args['seed'] = seed
    args['device'] = device

_set_random()
_set_device(args)

f = open(outfile, 'w')
print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)
model = DERNet(args, False)

for w in range(ww, 10):
    nl = (w+1)*10
    tmp = nl
    model.update_fc(nl)

    device = args['device'][0]
    model.load_state_dict(torch.load(f"weights/_{w}.pkl")['model_state_dict'])
    model.to(device)

    while tmp > 0:
        # load the dataset
        data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])
        test_dataset = data_manager.get_dataset(np.arange(tmp-10, tmp), source='test', mode='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


        # create the smooothed classifier g
        smoothed_classifier = Smooth(model, nl, sigma)


        # iterate through the dataset
        c1 = 0
        c2 = 0
        
        for i, (_, inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            # before_time = time()
            prediction, radius = smoothed_classifier.certify(inputs, N0, N, alpha, batch)
            # after_time = time()
            # correct = 0
            if prediction == targets and radius >= sigma:
                # correct = 1
                c1 += 1
            elif prediction == targets:
                c2 += 1
            # time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            # print("{}\t{}\t{}\t{}\t{}".format(i, targets, prediction, correct, time_elapsed), file=f, flush=True)
            # print(i, c1, c2)
        print(f"Final result {w} [{tmp-10}, {tmp}]: {c1} , {c2}", file=f, flush=True)
        tmp = tmp - 10
    
f.close()
