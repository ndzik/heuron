{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Forward where

import Control.Lens
import GHC.TypeLits
import Heuron.V1.Batched.Activation
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Layer
import Heuron.V1.Batched.Network
import Linear.Matrix

-- | Forward is the typelevel interpreter for our constructed network. It will
-- generate the correct amounts of forward calls for each layer and return the
-- updated network together with the final output.
class Forward as where
  forward :: as -> InputOf as -> (as, FinalOutputOf as)

-- | Recursion stop for Forward. Networks are at least two layers deep, input
-- and output layer.
instance
  {-# OVERLAPPING #-}
  ( KnownNat i,
    KnownNat n,
    KnownNat j,
    KnownNat k,
    KnownNat b,
    ActivationFunction af,
    ActivationFunction af',
    Compatible (Layer b i n af op) '[Layer b j k af' op']
  ) =>
  Forward (Network b '[Layer b i n af op, Layer b j k af' op'])
  where
  forward (pl :>: ll :>: NetworkEnd) i =
    let resPl = forwardInput pl i
        resLl = forwardInput ll resPl
     in (pl {_input = i} :>: ll {_input = resPl} :>: NetworkEnd, resLl)

-- | Recursion step for Forward. This will construct a chain of forward calls
-- that pass the output of the previous layer to the next layer.
instance
  ( Forward (Network b as),
    KnownNat i,
    KnownNat n,
    KnownNat b,
    KnownNat b',
    b ~ b',
    ActivationFunction af,
    Compatible (Layer b' i n af op) as,
    InputOf (Network b as) ~ Input b' n Double,
    FinalOutputOf (Network b as) ~ FinalOutputOf (Network b (Layer b' i n af op : as))
  ) =>
  Forward (Network b (Layer b' i n af op : as))
  where
  forward (l :>: ls) i =
    let resL = forwardInput l i
        (ls', res) = forward ls resL
     in (l {_input = i} :>: ls', res)

forwardInput ::
  (KnownNat n, KnownNat i, KnownNat b, ActivationFunction af) =>
  Layer b i n af op ->
  Input b i Double ->
  Input b n Double
forwardInput s inputs =
  -- This does for every neuron i (row) in the layer:
  -- > Σ(w_ij * x_j) + b_i
  -- where `i` is the neuron index and `j` is the input index.
  let transposedNeurons = s ^. weights . to transpose
      weightedInputs = (inputs !*! transposedNeurons) <&> (+ (s ^. bias))
      af = s ^. activationFunction . to activation
   in -- Apply activation and return result for this layer for each observation
      -- in input set.
      af weightedInputs
