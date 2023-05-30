{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Network
  ( ActivationFunction,

    -- * Layer & Lenses
    Layer (..),
    weights,
    bias,
    activation,

    -- * Network
    Network (..),
    (=|),

    -- * Forward propagation
    Forward (..),
  )
where

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

-- | Forward is the typelevel interpreter for our constructed network. It will
-- generate the correct amounts of forward calls for each layer.
class Forward as where
  forward :: as -> InputOf as -> OutputOf as

type family InputOf a where
  InputOf (Network b (Layer i n : as)) = Input b i Double

-- | OutputOf determines the final output of the given network depending on its
-- final layer.
type family OutputOf a where
  OutputOf (Network b '[Layer i n]) = Input b n Double
  OutputOf (Network b (Layer i n : as)) = OutputOf (Network b as)

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
    OutputOf (Network b as) ~ OutputOf (Network b (Layer i n : as))
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
