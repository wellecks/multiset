from collections import defaultdict
import visdom
import numpy as np
import matplotlib.pyplot as plt


class Visualizer(object):
    def __init__(self, output_directory, output_suffix):
        self.output_directory = output_directory
        self.output_suffix = output_suffix
        self._visdom = visdom.Visdom()
        self._visdom_panes = {}
        self._image_counters = defaultdict(int)
        plt.ioff()

    def step_entropies(self, entropies):
        # `entropies` is a vector of length num_steps
        counter = self._image_counters['step_entropies']
        self.plot_lines(entropies[np.newaxis, :], counter, 'step_entropies',
                        legend=map(str, np.arange(len(entropies)) + 1), )
        self._image_counters['step_entropies'] += 1

    def plot_lines(self, y, step, name, **opts):
        # `y` is N x num_lines`
        opts['title'] = opts.get('title', name)
        x = np.zeros(y.shape) + step
        if name in self._visdom_panes:
            self._visdom.updateTrace(x, y, win=self._visdom_panes[name],
                                     append=True,
                                     env=self.output_suffix)
        else:
            self._visdom_panes[name] = self._visdom.line(y, X=x,
                                                         opts=opts,
                                                         env=self.output_suffix)
