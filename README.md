# Heuron

This is a prototype implementation for describing neural networks in Haskell.
The basic idea is to have a backend agnostic DSL, which can have FPGAs, GPUs or CPUs as a target.

## Heuron.V1

V1 is an experiment of how I can achieve my goal of **correct by construction** neural networks.
For the meantime, V1 is a purely CPU based implementation using 100% Haskell code.

  * Heuron.V1.Single:
    - Initial API for describing a feed-forward neural net without backpropagation primitives
      on singular observations, i.e. batch size of 1. This will ultimately be removed together
      with all of V1, when V2 is realized.
  * Heuron.V1.Batched:
    - This contains a feed-forward neural net with training capabilities using backpropagation.
      The theory of how to realize and implement a neural net is not difficult, the complex
      part was how I could lift most of the construction into the type-level s.t. the compiler
      has all information available to prohibit the user from describing NNs that are:
        * unsupported
        * contain incompatible layers:
          * This includes forward & backward pass separately

```haskell
  let ann =
        inputLayer ReLU StochasticGradientDescent
          :>: hiddenLayer ReLU StochasticGradientDescent
          :>: hiddenLayer ReLU StochasticGradientDescent
          :>: hiddenLayer ReLU StochasticGradientDescent
            =| outputLayer Softmax StochasticGradientDescent
  -- > :t ann
  ann :: Network
    b
    '[Layer b 6 3 ReLU StochasticGradientDescent,
      Layer b 3 3 ReLU StochasticGradientDescent,
      Layer b 3 3 ReLU StochasticGradientDescent,
      Layer b 3 3 ReLU StochasticGradientDescent,
      Layer b 3 2 Softmax StochasticGradientDescent]
```

In the example we describe an ANN with three hidden layers. The input layer
expects 6 inputs and contains 3 neurons. `b` is the batchsize with which this
network will be trained. Since by the time of construction the batchsize might
be unknown it is left as an ambiguous type-parameter.
What is important to note is, that one can always ask the compiler to show a
description of ones neural network. Each layer can have its own activation
function (ReLU, Softmax, or whatever might be implement by a library user on
his own data-type) and optimizer (StochasticGradientDescent, etc.).

If the typeclasses are not implemented or the network description does not
adhere to certain constraints which guarantee correct networks, the compiler
will tell you something is wrong.

### Heuron.V1 - MNIST handwritten digits example

![Heuron-Net-Training](https://github.com/ndzik/heuron/assets/33512740/216066d3-19b5-45a8-88d1-28f6c610790f)

The executable defined by default uses the training set from the [MNIST database for handwritten digits](http://yann.lecun.com/exdb/mnist/).
Downloading the database and placing the training set in a `data/` folder within the directory
where `heuron` is started, will train a simple ANN on said dataset. This is a practical example
of `Heuron.V1` usage. The above picture draws the current network parameters during training.

The network defined is of the following type:

```haskell
ann :: Network
  batchSize
  '[Layer 100 pixelCount        hiddenNeuronCount ReLU    StochasticGradientDescent,
    Layer 100 hiddenNeuronCount hiddenNeuronCount ReLU    StochasticGradientDescent,
    Layer 100 hiddenNeuronCount hiddenNeuronCount ReLU    StochasticGradientDescent,
    Layer 100 hiddenNeuronCount 10                Softmax StochasticGradientDescent]
```

An ANN with a batchSize of `100`, an input layer expecting `pixelCount` inputs
containing `hiddenNeuronCount` neurons, using `ReLU` as its activation function
and `StochasticGradientDescent` as an optimizer.
The ANN has two hidden layers expecting `hiddenNeuronCount` inputs and containing `hiddenNeuronCount`
neurons using the `ReLU` activation function and `StochasticGradientDescent` as an optimizer.
The output layer expects `hiddenNeuronCount` inputs and contains `10` neurons using the
`Softmax` activation function to finally classify each digit, also using `StochasticGradientDescent`
as its optimizer.

### Heuron.V1 Layer description
Describing layers is rather easy. A few combinators are defined and more can be easily added.
The above example uses the following code to describe the layers:

```haskell
-- Describe network.
let learningRate = 0.25
inputLayer <- mkLayer $ do
  inputs @pixelCount
  neuronsWith @hiddenNeuronCount rng $ weightsScaledBy (1 / 784)
  activationF ReLU
  optimizerFunction (StochasticGradientDescent learningRate)

[hiddenLayer00, hiddenLayer01] <- mkLayers 2 $ do
  neuronsWith @hiddenNeuronCount rng $ weightsScaledBy (1 / 32)
  activationF ReLU
  optimizerFunction (StochasticGradientDescent learningRate)

outputLayer <- mkLayer $ do
  neurons @10 rng
  activationF Softmax
  optimizerFunction (StochasticGradientDescent learningRate)
```

* `inputs` allows to explicitly define the amount of inputs a layer is expecting.
* `neuronsWith` is required to set the number of neurons in this layer. `neurons` is a convenience
  function if there is no need to further modify the initial weightdistribution. 
* `activationF` allows to define the activation function.
* `optimizerFunction` sets the optimizer function.

Note how the hidden layers **do not** define their respective inputs. When the layers are used to
describe the ANN, GHC will automagically narrow the number of inputs down to the number of outputs
from the previous layer. One can, of course, still explicitly define the number of expected inputs
and if they do not match, GHC will tell you that somewith is wrong with your network description.

## Heuron.V2

With my experience from implementing V1 I want to generalize the created interfaces and make
them abstract enough to allow different net-generation backends. E.g. it should be possible
to let this library generate a GPU optimized neural net for training and a CPU/FPGA targeted
software net for execution. All with the same code.

## FAQ

**Q:** Why do you do this if there are things like TensorFlow, PyTorch, etc.?

**A:** I like types, I like proofs, I like static analysis. I like to learn stuff and build
       things from the ground up. I do, because I am.

**Q:** Is this possible with Haskell?

**A:** We will see.
