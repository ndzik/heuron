-- | Heuron.V1.Batched is a library for building neural networks. It is the
-- inital version of a purely CPU based neural network library utilizing the
-- GHC compiler to lift as much of the construction of the network to compile
-- time, thus creating a DSL that only allows "correct by construction" neural
-- networks.
--
-- The ultimate goal is to create a purely abstract neural network library that
-- is only concerned with the abstract definition of a network, while concrete
-- backends can utilize the GHC compiler to generate efficient code for
-- different backends and edge-case optimization (e.g. having a specific order
-- of optimizers/finalizers that can have their derivates fused together).
module Heuron.V1.Batched
  ( -- * Layer & Lenses
    Layer (..),
    weights,
    bias,

    -- * Network
    Network (..),
    (=|),

    -- * Network primitives
    reverseNetwork,
    module Heuron.V1.Batched.Input,

    -- * Forward propagation
    Forward (..),

    -- * Loss primitives
    categoricalCrossEntropy,

    -- * Activation primitives
    ActivationFunction (..),
    Differentiable (..),
    ReLU (..),
    Softmax (..),

    -- * Layer/Network optimizers
    module Heuron.V1.Batched.Optimizer,

    -- * Layer/Network backpropagation
    module Heuron.V1.Batched.Backprop,
  )
where

import Heuron.V1.Batched.Activation
import Heuron.V1.Batched.Backprop
import Heuron.V1.Batched.Forward
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Layer
import Heuron.V1.Batched.Loss
import Heuron.V1.Batched.Network
import Heuron.V1.Batched.Optimizer
