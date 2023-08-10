{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Network where

import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.V1.Batched.Activation
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Layer
import Linear.Matrix
import Linear.V
import Linear.Vector

-- | Network is an abstract definition of an artifical neural net. The batch
-- size for training is statically determined and given by its first
-- constructor argument.
--
-- Example:
-- @
--  let o1 = Layer inputLayerWeights inputLayerBias ReLU
--      o2 = Layer hidden01LayerWeights hidden01LayerBias ReLU
--      o3 = Layer hidden02LayerWeights hidden02LayerBias ReLU
--      o4 = Layer outputLayerWeights outputLayerBias ReLU
--      network = o1 :>: o2 :>: o3 :>: o4 :>: NetworkEnd
--      -- Alternatively using combinator syntax:
--      network' = o1 :>: o2 :>: o3 =| o4
--      result = forward network (head input)
-- @
data Network (b :: Nat) as where
  NetworkEnd :: Network b '[]
  (:>:) :: () => a -> Network b as -> Network b (a ': as)

type family Showable as :: Constraint where
  Showable '[] = ()
  Showable (a:as) = (Show a, Showable as)

instance (Showable net) => Show (Network b net) where
  show NetworkEnd = "=|"
  show (a :>: as) = unlines [show a ,":>:" ,show as]

infixr 5 :>:

infixr 5 =|

-- | (=|) is a used as a combinator to construct the output layer of a network.
(=|) :: a -> b -> Network c '[a, b]
(=|) = networkEnd

-- | networkEnd is a used as a combinator to construct a network.
networkEnd :: a -> b -> Network c '[a, b]
networkEnd a b = a :>: b :>: NetworkEnd

type family InputOf a where
  InputOf (Network b (Layer b i n _ _ : as)) = Input b i Double

type family Reversed as where
  Reversed (Network b as) = Network b (Reversed' as '[])

type family Reversed' as bs where
  Reversed' '[] bs = bs
  Reversed' (a ': as) bs = Reversed' as (a ': bs)

type family NextOutput a where
  NextOutput (Network b (Layer b i n _ _ : ls)) = Input b n Double

type family NextLayer as where
  NextLayer (Network b (Layer b i n af op : ls)) = Layer b i n af op

-- | FinalOutputOf determines the final output of the given network depending
-- on its final layer.
type family FinalOutputOf a where
  FinalOutputOf (Network b '[Layer b i n _ _]) = Input b n Double
  FinalOutputOf (Network b (Layer b i n _ _ : as)) = FinalOutputOf (Network b as)

-- | Compatible ensures that a given list of layers is compatible, which means
-- that the input dimensions for each layer are compatible with the output
-- dimension of each previous layer.
type family Compatible a bs :: Constraint where
  Compatible (Layer b i n af op) (Layer b' j k af' op' : bs) = (n ~ j, b ~ b', Compatible (Layer b' j k af' op') bs)
  Compatible (Layer b i n af op) '[] = ()

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
