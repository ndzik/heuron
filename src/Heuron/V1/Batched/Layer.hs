{-# LANGUAGE RoleAnnotations #-}

module Heuron.V1.Batched.Layer where

import Control.Lens
import Linear.V

type role Layer nominal nominal nominal representational representational

-- | Layers state, where n is the number of neurons and m is the number of
-- inputs.
data Layer (b :: k) (i :: k) (n :: k) af op = Layer
  { -- | Weights of the layer as a matrix of size m x n, where n is the number
    -- of neurons and m is the number of inputs. Each neuron is identified by
    -- the row index.
    _weights :: !(V n (V i Double)),
    -- | Bias of the layer as a vector of size n, where n is the number of
    -- neurons.
    _bias :: !(V n Double),
    -- | The cached input to this layer. Initialized to zero by default.
    _input :: !(V b (V i Double)),
    -- | The activation function used for each neuron in the layer.
    _activationFunction :: !af,
    -- | The optimizer used to adjust this layers weights and bias'.
    _optimizerFunction :: !op
  }

makeLenses ''Layer
