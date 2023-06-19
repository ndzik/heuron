module Heuron.V1.Batched.Optimizer where

import Data.Kind (Constraint)
import GHC.TypeLits (KnownNat, Nat)
import Heuron.V1.Batched.Layer
import Linear.Matrix
import Linear.V

type family CompatibleOptimizerParams a b c :: Constraint where
  CompatibleOptimizerParams (Layer b i n af op) (V n' (V i' Double)) (V n'' Double) = (i ~ i', n ~ n', n ~ n'')

type family NeuronNum f where
  NeuronNum (Layer b i n af _) = n

type family InputNum f where
  InputNum (Layer b i n af _) = i

class Optimizable f where
  optimize ::
    forall (n :: Nat) (i :: Nat).
    (n ~ NeuronNum f, i ~ InputNum f) =>
    f ->
    V n (V i Double) ->
    V n Double ->
    f

-- TODO: Implement the SGD optimizer.
instance Optimizable (Layer b i n af StochasticGradientDescent) where
  optimize l dWs dBs = l

data StochasticGradientDescent = StochasticGradientDescent

data StochasticGradientDescentMomentum = StochasticGradientDescentMomentum
