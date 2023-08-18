{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE RoleAnnotations #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Heuron.V1.Batched.Layer.Builder where

import Codec.Serialise
import Codec.Serialise.Decoding
import Control.Lens
import Control.Monad.Except
import Control.Monad.State
import Control.Monad.Writer (WriterT)
import Data.Data
import Data.Foldable
import Data.Monoid (Any (Any))
import Data.Vector (Vector)
import GHC.TypeLits
import Heuron.V1.Batched.Activation.Function (ActivationFunction, ReLU (..))
import Heuron.V1.Batched.Layer.Layer
import Heuron.V1.Batched.Optimizer.Types
import Heuron.V1.Matrix
import Heuron.V1.Vector
import Linear
import Linear.V
import System.Random.Stateful
import Text.Printf (printf)

data LayerBuilderError
  = NoActivationFunctionErr
  | NoOptimizerErr
  deriving (Show, Eq)

type LayerT (b :: k) (i :: k) (n :: k) af op m a = StateT (LayerBuilderState b i n af op) m a

data LayerBuilderState (b :: k) (i :: k) (n :: k) af op = LayerBuilderState
  { -- | Weights of the layer as a matrix of size m x n, where n is the number
    -- of neurons and m is the number of inputs. Each neuron is identified by
    -- the row index.
    _layerWeights :: !(V n (V i Double)),
    -- | Bias of the layer as a vector of size n, where n is the number of
    -- neurons.
    _layerBias :: !(V n Double),
    -- | The cached input to this layer. Initialized to zero by default.
    _layerInput :: !(V b (V i Double)),
    -- | The activation function used for each neuron in the layer.
    _layerAf :: !(Maybe af),
    -- | The optimizer used to adjust this layers weights and bias'.
    _layerOp :: !(Maybe op)
  }

makeLenses ''LayerBuilderState

mkLayers ::
  forall b i n af op m.
  (KnownNat b, KnownNat i, KnownNat n, ActivationFunction af, Monad m) =>
  Int ->
  LayerT b i n af op m () ->
  m [Layer b i n af op]
mkLayers n = replicateM n . mkLayer

mkLayer ::
  forall b i n af op m.
  (KnownNat b, KnownNat i, KnownNat n, ActivationFunction af, Monad m) =>
  LayerT b i n af op m () ->
  m (Layer b i n af op)
mkLayer builder = evalStateT (builder >> initialize) (LayerBuilderState zero zero zero Nothing Nothing)
  where
    initialize :: LayerT b i n af op m (Layer b i n af op)
    initialize = do
      af <- use layerAf >>= maybe (error "no activation function set") return
      op <- use layerOp >>= maybe (error "no optimizer set") return
      r <- get
      return $
        Layer
          { _weights = _layerWeights r,
            _bias = _layerBias r,
            _input = _layerInput r,
            _activationFunction = af,
            _optimizer = op
          }

inputs :: forall i b n af op m. (KnownNat i, Monad m) => LayerT b i n af op m ()
inputs = return ()

type LayerModifierT (n :: k) (i :: k) m a = StateT (LayerModifierState n i) m a

data LayerModifierState (n :: k) (i :: k) = LayerModifierState
  { _layerModifierWeights :: !(V n (V i Double)),
    _layerModifierBias :: !(V n Double)
  }

neuronsWith ::
  forall n i b af op g m r.
  (KnownNat n, KnownNat i, KnownNat b, RandomGen r, RandomGenM g r m) =>
  g ->
  LayerModifierT n i m () ->
  LayerT b i n af op m ()
neuronsWith rng modifier = do
  ws <- lift $ randomMS @n @i rng
  bs <- lift $ randomVS @n rng
  LayerModifierState ws bs <- lift $ execStateT modifier (LayerModifierState ws bs)
  modify $ \s -> s {_layerWeights = ws, _layerBias = bs}

weightsScaledBy :: (KnownNat n, KnownNat i, Monad m) => Double -> LayerModifierT n i m ()
weightsScaledBy s = modify $ \s' -> s' {_layerModifierWeights = fmap (* s) <$> _layerModifierWeights s'}

biasScaledBy :: (KnownNat n, Monad m) => Double -> LayerModifierT n i m ()
biasScaledBy s = modify $ \s' -> s' {_layerModifierBias = fmap (* s) (_layerModifierBias s')}

neurons ::
  forall n i b af op g m r.
  (KnownNat n, KnownNat i, KnownNat b, RandomGen r, RandomGenM g r m) =>
  g ->
  LayerT b i n af op m ()
neurons rng = neuronsWith rng $ return ()

activationF :: (Monad m, ActivationFunction af) => af -> LayerT b i n af op m ()
activationF af = modify $ \s -> s {_layerAf = Just af}

optimizerFunction :: (Monad m) => op -> LayerT b i n af op m ()
optimizerFunction op = modify $ \s -> s {_layerOp = Just op}

type LayerCombinator (b :: k) (i :: k) (n :: k) af op a = LayerT b i n af op IO a
