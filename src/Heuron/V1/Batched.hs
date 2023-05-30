module Heuron.V1.Batched
  ( ActivationFunction,

    -- * Layer & Lenses
    Layer (..),
    weights,
    bias,
    activation,

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
  )
where

import Heuron.V1.Batched.Forward
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Loss
import Heuron.V1.Batched.Network
