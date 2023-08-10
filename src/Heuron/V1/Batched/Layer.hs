{-# LANGUAGE RoleAnnotations #-}

module Heuron.V1.Batched.Layer where

import Control.Lens
import Data.Foldable
import Heuron.V1.Matrix (prettyMatrix)
import Heuron.V1.Vector (prettyVector)
import Linear.V
import Text.Printf (printf)

type role Layer nominal nominal nominal representational nominal

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
    _optimizer :: !op
  }

instance Show (Layer b i n af op) where
  show (Layer w b c _ _) = printf "Weights:\n%s\nBias:\n%s\nCached Input:%s" ws bs ci
    where
      ws = prettyMatrix w
      bs = prettyVector b
      ci = prettyMatrix c

makeLenses ''Layer
