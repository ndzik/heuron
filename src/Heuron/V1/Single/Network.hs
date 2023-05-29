{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Single.Network
  ( ActivationFunction,

    -- * Layer & Lenses
    Layer (..),
    weights,
    bias,
    activation,

    -- * Network
    Network (..),

    -- * Forward propagation
    Forward (..),
  )
where

import Control.Lens
import Data.Kind (Constraint)
import GHC.TypeLits
import Linear.Matrix
import Linear.V
import Linear.Vector

type ActivationFunction = Double -> Double

-- | Layers state, where n is the number of neurons and m is the number of
-- inputs.
data Layer (i :: k) (n :: k) = Layer
  { -- | Weights of the layer as a matrix of size m x n, where n is the number
    -- of neurons and m is the number of inputs. Each neuron is identified by
    -- the row index.
    _weights :: !(V n (V i Double)),
    -- | Bias of the layer as a vector of size n, where n is the number of
    -- neurons.
    _bias :: !(V n Double),
    -- | The activation function used for each neuron in the layer.
    _activation :: !ActivationFunction
  }

makeLenses ''Layer

infixr 5 :>:

-- | Network is an abstract definition of an artifical neural net.
--
-- Example:
-- @
--  let o1 = Layer inputLayerWeights inputLayerBias relu
--      o2 = Layer hidden01LayerWeights hidden01LayerBias relu
--      o3 = Layer hidden02LayerWeights hidden02LayerBias relu
--      o4 = Layer outputLayerWeights outputLayerBias relu
--      network = o1 :>: o2 :>: o3 :>: o4 :>: NetworkEnd
--      result = forward network (head input)
-- @
data Network as where
  NetworkEnd :: Network '[]
  (:>:) :: a -> Network as -> Network (a ': as)

-- | Forward is the typelevel interpreter for our constructed network. It will
-- generate the correct amounts of forward calls for each layer.
class Forward as where
  forward :: as -> InputOf as -> OutputOf as

type family InputOf a where
  InputOf (Network (Layer i n : as)) = V i Double

-- | OutputOf determines the final output of the given network depending on its
-- final layer.
type family OutputOf a where
  OutputOf (Network '[Layer i n]) = V n Double
  OutputOf (Network (Layer i n : as)) = OutputOf (Network as)

-- | Compatible ensures that a given list of layers is compatible, which means
-- that the input dimensions for each layer are compatible with the output
-- dimension of each previous layer.
type family Compatible a bs :: Constraint where
  Compatible (Layer i n) (Layer j k : bs) = (n ~ j, Compatible (Layer j k) bs)
  Compatible (Layer i n) '[] = ()

-- | Recursion stop for Forward. Networks are at least two layers deep, input
-- and output layer.
instance
  {-# OVERLAPPING #-}
  ( KnownNat i,
    KnownNat n,
    KnownNat j,
    KnownNat k,
    Compatible (Layer i n) '[Layer j k]
  ) =>
  Forward (Network '[Layer i n, Layer j k])
  where
  forward (pl :>: ll :>: NetworkEnd) = forwardInput ll . forwardInput pl

-- | Recursion step for Forward. This will construct a chain of forward calls
-- that pass the output of the previous layer to the next layer.
instance
  ( Forward (Network as),
    KnownNat i,
    KnownNat n,
    Compatible (Layer i n) as,
    InputOf (Network as) ~ V n Double,
    OutputOf (Network as) ~ OutputOf (Network (Layer i n : as))
  ) =>
  Forward (Network (Layer i n : as))
  where
  forward (l :>: ls) = forward ls . forwardInput l

forwardInput :: (KnownNat n, KnownNat i) => Layer i n -> V i Double -> V n Double
forwardInput s inputs = do
  -- This does for every neuron i (row) in the layer:
  -- > Σ(w_ij * x_j) + b_i
  -- where `i` is the neuron index and `j` is the input index.
  let weightedInputs = ((s ^. weights) !* inputs) ^+^ (s ^. bias)
      activationFunction = s ^. activation
  -- Apply activation and return result for this layer.
  activationFunction <$> weightedInputs
