# Heuron

To see more about the journey of this library and how it started, you can check out [JOURNEY](./JOURNEY.md).

# Current version Heuron.V2

V2 has its main focus on exploring how to define a DSL which allows pure, correct-by-construction definitions for networks that
can be translated to different backend code targeting whatever the heart desires.

## Describing a network

```haskell
generateV2Pytorch :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 16, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 60000) => IO ()
generateV2Pytorch = do
  rng <- newIOGenM (mkStdGen 42069)

  -- Describe network.
  let learningRate = 0.25
  inputLayer <- mkLayer @batchSize $ do
    inputs @'[pixelCount]
    neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 784)
    activationFunction ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  [hiddenLayer00] <- mkLayers 1 $ do
    neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 16)
    activationFunction ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  reshapeLayer <- Transform.mkLayer $ do
    Transform.output @'[28, 28, 28]

  [hiddenLayer01, hiddenLayer02] <- mkLayers 2 $ do
    activationFunction ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  resBlock <- Residual.mkBlock $ do
    Residual.activationFunction ReLU
    Residual.optimizerFunction (StochasticGradientDescent learningRate)
    inputLayer <- mkLayer $ do
      neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 32)
      activationFunction ReLU
      optimizerFunction (StochasticGradientDescent learningRate)

    [hiddenLayer00, hiddenLayer01, hiddenLayer02] <- mkLayers 3 $ do
      neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 16)
      activationFunction ReLU
      optimizerFunction (StochasticGradientDescent learningRate)

    dropL <- Drop.mkLayer 0.25

    outputLayer <- mkLayer $ do
      neurons @'[hiddenNeuronCount]
      activationFunction ReLU
      optimizerFunction (StochasticGradientDescent learningRate)

    return $ inputLayer :>: hiddenLayer00 :>: dropL :>: hiddenLayer01 :>: hiddenLayer02 :=> outputLayer

  outputLayer <- mkLayer $ do
    neurons @'[10]
    activationFunction Softmax
    optimizerFunction (StochasticGradientDescent learningRate)

  let ann =
        inputLayer
          :>: reshapeLayer
          :>: hiddenLayer01
          :>: resBlock
          :>: hiddenLayer00
          :=> outputLayer
      code = Torch.runPyTorch ann
  Torch.saveToModule "./torch_module.py" code
  print "PyTorch module written."
```

The example provided here uses the `PyTorch` backend, which just spits out a python module.

```python
import torch
import torch.nn as nn


class ResidualBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear((28, 28, 28), 16)
        self.fc1 = nn.Linear(16, 16)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 16)
        self.fc5 = nn.Linear(16, 16)

    def forward(self, x):
        residual = x
        x = self.fc0(x)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = x + residual
        x = torch.relu(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784, 16)
        self.fc1 = nn.Linear((28, 28, 28), (28, 28, 28))
        self.resblock2 = ResidualBlock2()
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.relu(x)
        # transform layer
        x = x.view((28, 28, 28))
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.resblock2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.softmax(x, dim=1)
        return x
```

I am still working and figuring out how to allow arbitrary definitions for _Layers reshaping inputs/outputs_.
The only mandatory thing is: A user should be able to inject arbitrary Layer types, effectively extending the DSL
accommodating his own needs.

Effectively: My guess is, everything can be descrsibed as some form of `Layer`, even arbitrary computation which
in itself does not need any backprop implementation (`Identity`). Still have to make sure that this is the case.

# Haskell.V3

V2 is proof enough for me to make this work. Thanks to [syedajafri1992's](https://www.reddit.com/r/haskell/comments/1l6ke0m/comment/mwufpqi/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) comment I will consider [HaskTorch](http://hasktorch.org/) in a backend translator, which let's this all stay Haskell.

## FAQ

**Q:** Why do you do this if there are things like TensorFlow, PyTorch, etc.?

**A:** I like types, I like proofs, I like static analysis. I like to learn stuff and build
       things from the ground up. I do, because I am.

**Q:** Is this possible with Haskell?

**A:** We will see.
