{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Forward where

import Control.Lens
import GHC.TypeLits
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Network
import Linear.Matrix

-- | Forward is the typelevel interpreter for our constructed network. It will
-- generate the correct amounts of forward calls for each layer.
class Forward as where
  forward :: as -> InputOf as -> FinalOutputOf as

-- | Recursion stop for Forward. Networks are at least two layers deep, input
-- and output layer.
instance
  {-# OVERLAPPING #-}
  ( KnownNat i,
    KnownNat n,
    KnownNat j,
    KnownNat k,
    KnownNat b,
    Compatible (Layer i n) '[Layer j k]
  ) =>
  Forward (Network b '[Layer i n, Layer j k])
  where
  forward (pl :>: ll :>: NetworkEnd) i = forwardInput ll . forwardInput pl $ i

-- | Recursion step for Forward. This will construct a chain of forward calls
-- that pass the output of the previous layer to the next layer.
instance
  ( Forward (Network b as),
    KnownNat i,
    KnownNat n,
    KnownNat b,
    Compatible (Layer i n) as,
    InputOf (Network b as) ~ Input b n Double,
    FinalOutputOf (Network b as) ~ FinalOutputOf (Network b (Layer i n : as))
  ) =>
  Forward (Network b (Layer i n : as))
  where
  forward (l :>: ls) = forward ls . forwardInput l

forwardInput :: (KnownNat m, KnownNat n, KnownNat i) => Layer i n -> Input m i Double -> Input m n Double
forwardInput s inputs =
  -- This does for every neuron i (row) in the layer:
  -- > Σ(w_ij * x_j) + b_i
  -- where `i` is the neuron index and `j` is the input index.
  let transposedNeurons = s ^. weights . to transpose
      weightedInputs = (inputs !*! transposedNeurons) <&> (+ (s ^. bias))
      activationFunction = s ^. activation
   in -- Apply activation and return result for this layer for each observation
      -- in input set.
      (activationFunction <$>) <$> weightedInputs
