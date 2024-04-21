import random
from littlegrad.engine import Value

class Neuron:

  def __init__(self, n_in, non_lin=True):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
    self.b = Value(random.uniform(-1, 1))
    self.non_lin = non_lin

  def __call__(self, x):
    act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
    return act.relu() if self.non_lin else act

  def parameters(self):
    return self.w + [self.b]

  def __repr__(self):
    return f"{'ReLU' if self.non_lin else 'Linear'}Neuron({len(self.w)})"

class Layer:

  def __init__(self, n_in, n_out, **kwargs):
    self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

  def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP:

  def __init__(self, n_in, n_outs):
    sz = [n_in] + n_outs
    self.layers = [Layer(sz[i], sz[i+1], non_lin=i!=len(n_outs)-1) for i in range(len(n_outs))]

  def __call__(self, x):
    for l in self.layers:
      x = l(x)
    return x

  def parameters(self):
    return [p for l in self.layers for p in l.parameters()]

  def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
