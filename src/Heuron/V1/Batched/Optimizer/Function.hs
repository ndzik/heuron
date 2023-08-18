{-# LANGUAGE DeriveGeneric #-}

module Heuron.V1.Batched.Optimizer.Function where

import Codec.Serialise
import Control.Lens ((^.))
import Data.Kind (Constraint)
import GHC.Generics
import GHC.TypeLits (KnownNat, Nat)
import Heuron.V1.Batched.Layer
import Heuron.V1.Batched.Optimizer.Types
import Linear ((*^))
import Linear.Matrix
import Linear.V
import Linear.Vector ((^-^))

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

instance (KnownNat b, KnownNat i, KnownNat n) => Optimizable (Layer b i n af StochasticGradientDescent) where
  optimize l dWs dBs = l {_weights = newWeights, _bias = newBiases}
    where
      StochasticGradientDescent learningRate = l ^. optimizer
      ws = l ^. weights
      bs = l ^. bias
      newWeights = ws !-! (fmap (learningRate *) <$> dWs)
      newBiases = bs ^-^ (learningRate *^ dBs)

-- | The StochasticGradientDescent optimizer with learning rate.
newtype StochasticGradientDescent = StochasticGradientDescent Double
  deriving (Generic, Show, Eq)

instance Serialise StochasticGradientDescent

data StochasticGradientDescentMomentum = StochasticGradientDescentMomentum
  deriving (Generic, Show, Eq)

instance Serialise StochasticGradientDescentMomentum
