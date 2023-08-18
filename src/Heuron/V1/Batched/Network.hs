{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Network where

import Codec.Serialise
import Codec.Serialise.Decoding
import Codec.Serialise.Encoding
import Control.Lens
import Data.Data (Proxy (..))
import Data.Kind (Constraint)
import Data.Vector (Vector)
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
  (:>:) :: () => Layer b i n af op -> Network b as -> Network b (Layer b i n af op ': as)

type family Showable as :: Constraint where
  Showable '[] = ()
  Showable (a : as) = (Show a, Showable as)

instance (Showable net) => Show (Network b net) where
  show NetworkEnd = "=|"
  show (a :>: as) = unlines [show a, ":>:", show as]

type family ComparableNetwork net :: Constraint where
  ComparableNetwork '[] = ()
  ComparableNetwork (a ': as) = (Eq a, ComparableNetwork as)

instance (ComparableNetwork net) => Eq (Network b net) where
  NetworkEnd == NetworkEnd = True
  (a :>: as) == (b :>: bs) = a == b && as == bs

type family SerializableNetwork net :: Constraint where
  SerializableNetwork (Network b '[]) = ()
  SerializableNetwork (Network b (a ': as)) = (Serialise a, SerializableNetwork (Network b as))

instance Serialise (Network b '[]) where
  encode NetworkEnd = mempty
  decode = pure NetworkEnd

instance (Serialise l, KnownNat b, Serialise (Network b ls), l ~ Layer b i n af op, SerializableNetwork (Network b ls)) => Serialise (Network b (l ': ls)) where
  encode (l :>: ls) = encode l <> encode ls

  decode :: forall s b i n af op ls. (l ~ Layer b i n af op, Serialise (Network b ls)) => Decoder s (Network b (l ': ls))
  decode = do
    l <- decode @(Layer b i n af op)
    ls <- decode @(Network b ls)
    pure $ l :>: ls

infixr 5 :>:

infixr 5 =|

-- | (=|) is a used as a combinator to construct the output layer of a network.
(=|) ::
  ( KnownNat i,
    KnownNat n,
    KnownNat c,
    KnownNat i',
    KnownNat n',
    a ~ Layer c i n af op,
    b ~ Layer c i' n' af' op'
  ) =>
  a ->
  b ->
  Network c '[a, b]
(=|) = networkEnd

-- | networkEnd is a used as a combinator to construct a network.
networkEnd ::
  ( KnownNat i,
    KnownNat n,
    KnownNat c,
    KnownNat i',
    KnownNat n',
    a ~ Layer c i n af op,
    b ~ Layer c i' n' af' op'
  ) =>
  a ->
  b ->
  Network c '[a, b]
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

type family MadeOfLayers net net' :: Constraint where
  MadeOfLayers (a : ls') (Layer b i n af op : ls) = (a ~ Layer b i n af op, MadeOfLayers ls' ls)
  MadeOfLayers '[] '[] = ()

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

class IteratableNetwork n where
  forLayerIn ::
    (Monoid a) =>
    n ->
    -- | The function to apply to each layer:
    --  (batch size, input dimension, number of neurons) -> weights -> bias -> a
    ((Int, Int, Int) -> Vector (Vector Double) -> Vector Double -> a) ->
    a

instance IteratableNetwork (Network b '[]) where
  forLayerIn NetworkEnd _ = mempty

instance (KnownNat b, KnownNat i, KnownNat n, IteratableNetwork (Network b ls)) => IteratableNetwork (Network b (Layer b i n af op ': ls)) where
  forLayerIn (l :>: ls) f =
    f
      (fromIntegral $ natVal (Proxy @b), fromIntegral $ natVal (Proxy @i), fromIntegral $ natVal (Proxy @n))
      (l ^. weights . to (toVector . fmap toVector))
      (l ^. bias . to toVector)
      <> forLayerIn ls f
