{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

module Heuron.V1.Network where

-- | Network describes a neural network. `Network Input Output [Hidden]`.
data Network = Network !Layer !Layer ![Layer]

-- | A NetworkInterpreter is responsible for interpreting the described neural
-- network. It encapsulates a backend implementation. The implementation of the
-- interpreter decides what kind (TODO: or type?) the activation function have
-- and how the resulting code looks like.
--
-- E.g. a haskell software backend will require the ActivationFunction to be
-- of type `(Inputs, Weights) -> Bias -> a`, where a is dependent on the
-- specific activation function used.
-- Furthermore we want to be able to write a generic trainer and executor for
-- the network using any backend.
--
-- Training:
--  * Feed forward data -> Evaluate layers (step) -> Result
--  * Backprop and let network update
--
-- Having a generic way to train the network independent of the backend allows
-- implementation of visualization methods for viewing the training progress.
class NetworkInterpreter i

-- | A layer of a neural network with n neurons utilizing the given
-- activation/classifier function.
data Layer where
  MkLayer :: Int -> f -> Layer

-- -- | ActivationFunction describes an activation function. In context to a neural
-- -- network it is required to be differentiable.
-- class ActivationFunction f where
--   -- | activate uses the given activation function `f` together with a
--   -- collection of inputs, collection of weights and a bias to return an
--   -- activation result.
--   activate :: Functor c => f -> c Double -> c Double -> Double -> Double

type family Activation f

-- | Use an open type-family to allow injection of new activation functions
-- outside the scope of this module/library.
-- data family Activation f

-- | Softmax activation function.
data Softmax = Softmax

-- | ReLU activation function.
data ReLU = ReLU
