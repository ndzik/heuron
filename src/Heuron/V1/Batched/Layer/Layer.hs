{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Heuron.V1.Batched.Layer.Layer where

import Codec.Serialise
import Codec.Serialise.Decoding
import Control.Lens
import Control.Monad.State
import Control.Monad.Writer (WriterT)
import Data.Data
import Data.Foldable
import Data.Monoid (Any (Any))
import Data.Vector (Vector)
import GHC.TypeLits
import Heuron.V1.Batched.Activation.Function (ActivationFunction, ReLU (..))
import Heuron.V1.Batched.Optimizer.Types
import Heuron.V1.Matrix (prettyMatrix)
import Heuron.V1.Vector (prettyVector)
import Linear
import Linear.V
import Text.Printf (printf)

type role Layer nominal nominal nominal representational nominal

-- | Layers state, where b is the number samples per batch, i the number of
-- inputs and n the number of neurons.
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
  deriving (Eq)

instance (KnownNat b, KnownNat i, KnownNat n, Serialise af, Serialise op) => Serialise (Layer b i n af op) where
  encode (Layer w b c _ _) = encode (toVector . fmap toVector $ w, toVector b, toVector . fmap toVector $ c)

  decode ::
    forall b i n af op s.
    ( KnownNat b,
      KnownNat i,
      KnownNat n,
      Serialise af,
      Serialise op
    ) =>
    Decoder s (Layer b i n af op)
  decode = do
    (wV, bV, cV) <- decode @(Vector (Vector Double), Vector Double, Vector (Vector Double))
    w <- case toMatrix @n @i wV of
      Nothing -> fail "Could not decode weight matrix"
      Just w -> pure w
    b <- case fromVector @n bV of
      Nothing -> fail "Could not decode bias vector"
      Just b -> pure b
    c <- case toMatrix @b @i cV of
      Nothing -> fail "Could not decode cached input vector"
      Just c -> pure c
    af <- decode @af
    op <- decode @op
    pure $ Layer w b c af op
    where
      toMatrix :: forall n i. (Dim n, Dim i) => Vector (Vector Double) -> Maybe (V n (V i Double))
      toMatrix mv = ((fromVector @n) mv <&> fmap (fromVector @i)) >>= sequence

instance Show (Layer b i n af op) where
  show (Layer w b c _ _) = printf "Weights:\n%s\nBias:\n%s\nCached Input:%s" ws bs ci
    where
      ws = prettyMatrix w
      bs = prettyVector b
      ci = prettyMatrix c

makeLenses ''Layer
