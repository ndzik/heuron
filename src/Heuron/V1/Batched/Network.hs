{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Network where

import Control.Lens
import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.V1.Batched.Input
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

-- | Network is an abstract definition of an artifical neural net. The batch
-- size for training is statically determined and given by its first
-- constructor argument.
--
-- Example:
-- @
--  let o1 = Layer inputLayerWeights inputLayerBias relu
--      o2 = Layer hidden01LayerWeights hidden01LayerBias relu
--      o3 = Layer hidden02LayerWeights hidden02LayerBias relu
--      o4 = Layer outputLayerWeights outputLayerBias relu
--      network = o1 :>: o2 :>: o3 :>: o4 :>: NetworkEnd
--      -- Alternatively using combinator syntax:
--      network' = o1 :>: o2 :>: o3 =| o4
--      result = forward network (head input)
-- @
data Network b as where
  NetworkEnd :: Network b '[]
  (:>:) :: a -> Network b as -> Network b (a ': as)

infixr 5 =|

-- | (=|) is a used as a combinator to construct the output layer of a network.
(=|) :: a -> b -> Network c '[a, b]
(=|) = networkEnd

-- | networkEnd is a used as a combinator to construct a network.
networkEnd :: a -> b -> Network c '[a, b]
networkEnd a b = a :>: b :>: NetworkEnd

type family InputOf a where
  InputOf (Network b (Layer i n : as)) = Input b i Double

type family Reversed as where
  Reversed (Network b as) = Network b (Reversed' as '[])

type family Reversed' as bs where
  Reversed' '[] bs = bs
  Reversed' (a ': as) bs = Reversed' as (a ': bs)

-- | FinalOutputOf determines the final output of the given network depending
-- on its final layer.
type family FinalOutputOf a where
  FinalOutputOf (Network b '[Layer i n]) = Input b n Double
  FinalOutputOf (Network b (Layer i n : as)) = FinalOutputOf (Network b as)

-- | Compatible ensures that a given list of layers is compatible, which means
-- that the input dimensions for each layer are compatible with the output
-- dimension of each previous layer.
type family Compatible a bs :: Constraint where
  Compatible (Layer i n) (Layer j k : bs) = (n ~ j, Compatible (Layer j k) bs)
  Compatible (Layer i n) '[] = ()

type family IsReversed as bs cs :: Constraint where
  IsReversed (a : as) '[] cs = (IsReversed as '[a] cs)
  IsReversed (a : as) bs cs = (IsReversed as (a ': bs) cs)
  IsReversed '[] bs cs = (bs ~ cs)

-- | Reverses the order of layers for the given network.
--
-- E.g.:
--
--  > let ann = inputLayer id :>: hiddenLayer id =| outputLayer id
--  >     ann' = reverseNetwork ann
--
--  > :type ann
--  > ann :: Network b '[Layer 6 2, Layer 3 3, Layer 3 2]
--
--  > :type ann'
--  > ann' :: Network b '[Layer 3 2, Layer 3 3, Layer 6 2]
reverseNetwork ::
  (IsReversed as '[] cs, cs ~ Reversed' as '[]) =>
  Network b as ->
  Network b cs
reverseNetwork nn = reverse' nn NetworkEnd
  where
    reverse' :: (IsReversed ds es fs) => Network b ds -> Network b es -> Network b fs
    reverse' NetworkEnd acc = acc
    reverse' (l :>: ls) acc = reverse' ls (l :>: acc)
