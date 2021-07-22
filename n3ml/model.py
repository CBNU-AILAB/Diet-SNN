import math
import torch
import torch.nn as nn
import torch.distributions.uniform

from n3ml.layer import IF1d, IF2d, Conv2d, AvgPool2d, Linear, Bohte, TravanaeiAndMaida

import n3ml.network
import n3ml.layer
import n3ml.population
import n3ml.connection
import n3ml.learning


class Voelker2015(n3ml.network.Network):
    def __init__(self,
                 neurons: int = 100,
                 input_size: int = 784,
                 output_size: int = 10):
        super().__init__()
        self.add_component('pop', n3ml.population.NEF(neurons=neurons,
                                                      input_size=input_size,
                                                      output_size=output_size,
                                                      neuron_type=n3ml.population.LIF))

    def init_vars(self) -> None:
        for p in self.population.values():
            p.init_vars()

    def init_params(self) -> None:
        for p in self.population.values():
            p.init_params()


class Wu2018(n3ml.network.Network):
    def __init__(self, batch_size, time_interval):
        super().__init__()
        self.conv1         = nn.Conv2d(1, 32,  kernel_size=3, stride=1, padding=1)
        self.conv2         = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1           = nn.Linear(7 * 7 * 32, 128)
        self.fc2           = nn.Linear(128, 10)
        self.avgpool       = nn.AvgPool2d(kernel_size=2)
        self.batch_size    = batch_size
        self.time_interval = time_interval

    def mem_update(self, ops, x, mem, spike):
        mem = mem * 0.2 * (1. - spike) + ops(x)
        STBP_spike = n3ml.layer.Wu()
        spike      = STBP_spike(mem)
        return mem, spike

    def forward(self, input):
        c1_mem = c1_spike = torch.zeros(self.batch_size, 32, 28, 28).cuda()
        c2_mem = c2_spike = torch.zeros(self.batch_size, 32, 14, 14).cuda()

        h1_mem = h1_spike               = torch.zeros(self.batch_size, 128).cuda()
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, 10).cuda()

        for time in range(self.time_interval):

            x = input > torch.rand(input.size()).cuda()

            c1_mem, c1_spike = self.mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            x = self.avgpool(c1_spike)

            c2_mem, c2_spike = self.mem_update(self.conv2,x, c2_mem,c2_spike)
            x = self.avgpool(c2_spike)

            x = x.view(self.batch_size, -1)

            h1_mem, h1_spike = self.mem_update(self.fc1, x, h1_mem, h1_spike)
            h2_mem, h2_spike = self.mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / self.time_interval

        return outputs


class Ponulak2005(n3ml.network.Network):
    def __init__(self,
                 neurons: int = 800,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.neurons = neurons
        self.num_classes = num_classes
        self.add_component('input', n3ml.population.Input(1*28*28,
                                                          traces=False))
        self.add_component('hidden', n3ml.population.LIF(neurons,
                                                         tau_ref=2.0,
                                                         traces=False,
                                                         rest=0.0,
                                                         reset=0.0,
                                                         v_th=1.0,
                                                         tau_rc=10.0))
        self.add_component('output', n3ml.population.LIF(num_classes,
                                                         tau_ref=2.0,
                                                         traces=False,
                                                         rest=0.0,
                                                         reset=0.0,
                                                         v_th=1.0,
                                                         tau_rc=10.0))
        self.add_component('ih', n3ml.connection.Synapse(self.input, self.hidden))
        self.add_component('ho', n3ml.connection.Synapse(self.hidden, self.output))

    def reset_parameters(self):
        for synapse in self.connection.values():
            synapse.w[:] = torch.rand_like(synapse.w) - 0.5


class DiehlAndCook2015(n3ml.network.Network):
    def __init__(self, neurons: int = 100):
        super().__init__()
        self.neurons = neurons
        self.add_component('inp', n3ml.population.Input(1*28*28,
                                                        traces=True,
                                                        tau_tr=20.0))
        self.add_component('exc', n3ml.population.DiehlAndCook(neurons,
                                                               traces=True,
                                                               rest=-65.0,
                                                               reset=-60.0,
                                                               v_th=-52.0,
                                                               tau_ref=5.0,
                                                               tau_rc=100.0,
                                                               tau_tr=20.0))
        self.add_component('inh', n3ml.population.LIF(neurons,
                                                      traces=False,
                                                      rest=-60.0,
                                                      reset=-45.0,
                                                      v_th=-40.0,
                                                      tau_rc=10.0,
                                                      tau_ref=2.0,
                                                      tau_tr=20.0))
        self.add_component('xe', n3ml.connection.LinearSynapse(self.inp,
                                                               self.exc,
                                                               alpha=78.4,
                                                               learning_rule=n3ml.learning.PostPre,
                                                               initializer=torch.distributions.uniform.Uniform(0, 0.3)))
        self.add_component('ei', n3ml.connection.LinearSynapse(self.exc,
                                                               self.inh,
                                                               w_min=0.0,
                                                               w_max=22.5))
        self.add_component('ie', n3ml.connection.LinearSynapse(self.inh,
                                                               self.exc,
                                                               w_min=-120.0,
                                                               w_max=0.0))

        # Initialize synaptic weight for each synapse
        self.xe.init()
        self.ei.w[:] = torch.diagflat(torch.ones_like(self.ei.w)[0] * 22.5)
        self.ie.w[:] = (torch.ones_like(self.ie.w) * -120.0).fill_diagonal_(0.0)


class Hunsberger2015(n3ml.network.Network):
    def __init__(self, amplitude, tau_ref, tau_rc, gain, sigma, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.AvgPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 1024, bias=False),
            n3ml.layer.SoftLIF(amplitude=amplitude, tau_ref=tau_ref, tau_rc=tau_rc, gain=gain, sigma=sigma),
            nn.Linear(1024, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x


class Bohte2002(n3ml.network.Network):
    def __init__(self) -> None:
        super().__init__()

        self.add_component('fc1', Bohte(50, 10, time_constant=7.0))
        self.add_component('fc2', Bohte(10, 3, time_constant=7.0))
        # self.add_component('fc', Bohte(50, 3))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: 현재 시점
        # x: 스파이크 발화 시점에 대한 정보
        x = self.fc1(t, x)
        x = self.fc2(t, x)
        # x = self.fc(t, x)
        return x


class TravanaeiAndMaida2017(n3ml.network.Network):
    def __init__(self,
                 num_classes: int = 10,
                 hidden_neurons: int = 100) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_neurons = hidden_neurons
        self.add_component('fc1', TravanaeiAndMaida(in_neurons=1*28*28,
                                                    out_neurons=hidden_neurons,
                                                    threshold=0.9))
        self.add_component('fc2', TravanaeiAndMaida(in_neurons=hidden_neurons,
                                                    out_neurons=num_classes,
                                                    threshold=hidden_neurons*0.025))

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        o = self.fc1(o)
        o = self.fc2(o)
        return o

    def reset_variables(self, **kwargs):
        for l in self.layer.values():
            l.reset_variables(**kwargs)


class Cao2015_Tailored(n3ml.network.Network):
    def __init__(self,
                 num_classes: int = 10,
                 in_planes: int = 3,
                 out_planes: int = 64) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.extractor = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_planes, out_planes, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(out_planes, out_planes, 3, bias=False),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_planes, out_planes, bias=False),
            nn.ReLU(),
            nn.Linear(out_planes, num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Cao2015_SNN(n3ml.network.Network):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class Ho2013(n3ml.network.Network):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TailoredCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, out_channels=64):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.extractor = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(self.out_channels, self.out_channels, 5, bias=False),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(self.out_channels, self.out_channels, 3, bias=False),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.out_channels, self.num_classes, bias=False)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class Diet2020_ANN(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout     = dropout
        self.extractor   = nn.Sequential(
            nn.Conv2d(  3,  64, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d( 64,  64, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout))

        self.classifier  = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096, bias=False), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 10, bias=False))

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Diet2020_SNN(nn.Module):
    def __init__(self, labels=10, timesteps=10, leak=1.0, default_threshold=1.0, dropout=0.2, kernel_size=3):
        super().__init__()
        self.labels = labels
        self.timesteps = timesteps
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.vmem_drop = 0
        self.mem = {}
        self.mask = {}
        self.spike = {}
        self.extractor   = nn.Sequential(
            nn.Conv2d(  3,  64, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d( 64,  64, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1, bias=False), nn.ReLU(inplace=True), nn.Dropout(self.dropout))

        self.classifier  = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096, bias=False), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096, bias=False), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 10, bias=False))

        self.initialize_weights()

        threshold = {}
        lk = {}
        width = 32
        height = 32

        for l in range(len(self.extractor)):
            if isinstance(self.extractor[l], nn.Conv2d):
                threshold['t' + str(l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(l)] = nn.Parameter(torch.tensor(leak))

            elif isinstance(self.extractor[l], nn.AvgPool2d):
                width = width // self.extractor[l].kernel_size
                height = height // self.extractor[l].kernel_size

        prev = len(self.extractor)
        for l in range(len(self.classifier) - 1):
            if isinstance(self.classifier[l], nn.Linear):
                threshold['t' + str(prev + l)] = nn.Parameter(torch.tensor(default_threshold))
                lk['l' + str(prev + l)] = nn.Parameter(torch.tensor(leak))

        self.threshold = nn.ParameterDict(threshold)
        self.leak = nn.ParameterDict(lk)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def threshold_update(self, scaling_factor=1.0, thresholds=[]):
        self.scaling_factor = scaling_factor
        width = 32
        height = 32

        for pos in range(len(self.extractor)):
            if isinstance(self.extractor[pos], nn.Conv2d):
                if thresholds:
                    self.threshold.update({'t' + str(pos): nn.Parameter(torch.tensor(thresholds.pop(0)) * self.scaling_factor)})
            elif isinstance(self.extractor[pos], nn.AvgPool2d):
                width = width // self.extractor[pos].kernel_size
                height = height // self.extractor[pos].kernel_size

        prev = len(self.extractor)

        for pos in range(len(self.classifier) - 1):
            if isinstance(self.classifier[pos], nn.Linear):
                if thresholds:
                    self.threshold.update({'t' + str(prev + pos): nn.Parameter(torch.tensor(thresholds.pop(0)) * self.scaling_factor)})

    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width = x.size(2)
        self.height = x.size(3)
        self.mem = {}
        self.spike = {}
        self.mask = {}

        for l in range(len(self.extractor)):
            if isinstance(self.extractor[l], nn.Conv2d):
                self.mem[l] = torch.zeros(self.batch_size, self.extractor[l].out_channels, self.width, self.height)

            elif isinstance(self.extractor[l], nn.ReLU):
                if isinstance(self.extractor[l - 1], nn.Conv2d):
                    self.spike[l] = torch.ones(self.mem[l - 1].shape) * (-1000)
                elif isinstance(self.extractor[l - 1], nn.AvgPool2d):
                    self.spike[l] = torch.ones(self.batch_size, self.extractor[l - 2].out_channels, self.width, self.height) * (-1000)

            elif isinstance(self.extractor[l], nn.Dropout):
                self.mask[l] = self.extractor[l](torch.ones(self.mem[l - 2].shape).cuda())

            elif isinstance(self.extractor[l], nn.AvgPool2d):
                self.width = self.width // self.extractor[l].kernel_size
                self.height = self.height // self.extractor[l].kernel_size

        prev = len(self.extractor)

        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev + l] = torch.zeros(self.batch_size, self.classifier[l].out_features)

            elif isinstance(self.classifier[l], nn.ReLU):
                self.spike[prev + l] = torch.ones(self.mem[prev + l - 1].shape) * (-1000)

            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[prev + l] = self.classifier[l](torch.ones(self.mem[prev + l - 2].shape).cuda())

    def forward(self, x, find_max_mem=False, max_mem_layer=0, percentile=99.7):
        self.neuron_init(x)
        max_mem = 0.0

        for t in range(self.timesteps):
            out_prev = x
            for l in range(len(self.extractor)):
                if isinstance(self.extractor[l], (nn.Conv2d)):
                    if find_max_mem and l == max_mem_layer:
                        cur = self.percentile(self.extractor[l](out_prev).view(-1), percentile)
                        if (cur > max_mem):
                            max_mem = torch.tensor([cur])
                        break

                    delta_mem = self.extractor[l](out_prev)
                    self.mem[l] = getattr(self.leak, 'l' + str(l)) * self.mem[l] + delta_mem
                    mem_thr = (self.mem[l] / getattr(self.threshold, 't' + str(l))) - 1.0
                    rst = getattr(self.threshold, 't' + str(l)) * (mem_thr > 0).float()
                    self.mem[l] = self.mem[l] - rst

                elif isinstance(self.extractor[l], nn.ReLU):
                    STDB_spike = n3ml.layer.Rathi()
                    out = STDB_spike(mem_thr)
                    self.spike[l] = self.spike[l].masked_fill(out.bool(), t - 1)
                    out_prev = out.clone()

                elif isinstance(self.extractor[l], nn.AvgPool2d):
                    out_prev = self.extractor[l](out_prev)

                elif isinstance(self.extractor[l], nn.Dropout):
                    out_prev = out_prev * self.mask[l]

            if find_max_mem and max_mem_layer < len(self.extractor):
                continue

            out_prev = out_prev.reshape(self.batch_size, -1)
            prev = len(self.extractor)

            for l in range(len(self.classifier) - 1):
                if isinstance(self.classifier[l], (nn.Linear)):
                    if find_max_mem and (prev + l) == max_mem_layer:
                        cur = self.percentile(self.classifier[l](out_prev).view(-1), percentile)
                        if cur > max_mem:
                            max_mem = torch.tensor([cur])
                        break

                    delta_mem = self.classifier[l](out_prev)
                    self.mem[prev + l] = getattr(self.leak, 'l' + str(prev + l)) * self.mem[prev + l] + delta_mem
                    mem_thr = (self.mem[prev + l] / getattr(self.threshold, 't' + str(prev + l))) - 1.0
                    rst = getattr(self.threshold, 't' + str(prev + l)) * (mem_thr > 0).float()
                    self.mem[prev + l] = self.mem[prev + l] - rst

                elif isinstance(self.classifier[l], nn.ReLU):
                    STDB_spike = n3ml.layer.Rathi()
                    out = STDB_spike(mem_thr)
                    self.spike[prev + l] = self.spike[prev + l].masked_fill(out.bool(), t - 1)
                    out_prev = out.clone()

                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev = out_prev * self.mask[prev + l]

            # Compute the classification layer outputs
            if not find_max_mem:
                self.mem[prev + l + 1] = self.mem[prev + l + 1] + self.classifier[l + 1](out_prev)
        if find_max_mem:
            return max_mem

        return self.mem[prev + l + 1]