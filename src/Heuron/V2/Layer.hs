module Heuron.V2.Layer where

import Control.Lens
import Control.Monad.State
import Data.Default
import Data.Maybe
import GHC.TypeLits

type LayerSpec = *

data family Layer (b :: Nat) (i :: Nat) (n :: Nat) (l :: LayerSpec)

data LinearLayer (i :: Nat) (n :: Nat) af op = LinearLayer af op [ModifierAction i n]

newtype instance Layer (b :: Nat) (i :: Nat) (n :: Nat) (LinearLayer i n af op) = Linear (LinearLayer i n af op)

type LayerT (i :: Nat) (n :: Nat) af op m a = StateT (LayerBuilderState i n af op) m a

data ModifierAction (i :: Nat) (n :: Nat) = ScaleWeights Double | ScaleBias Double

data LayerBuilderState (i :: Nat) (n :: Nat) af op = LayerBuilderState
  { _layerModifierActions :: ![ModifierAction i n],
    _layerAf :: !(Maybe af),
    _layerOp :: !(Maybe op)
  }

instance Default (LayerBuilderState i n af op) where
  def = LayerBuilderState [] Nothing Nothing

makeLenses ''LayerBuilderState

mkLayers ::
  forall n b af op m.
  (KnownNat n, Monad m) =>
  Int ->
  LayerT n n af op m () ->
  m [Layer b n n (LinearLayer n n af op)]
mkLayers n = replicateM n . mkLayer

mkLayer :: forall i b n m af op. (KnownNat i, KnownNat n, Monad m) => LayerT i n af op m () -> m (Layer b i n (LinearLayer i n af op))
mkLayer builder = evalStateT (builder >> initialize) def
  where
    initialize = do
      af <- use layerAf >>= maybe (error "no activation function set") pure
      op <- use layerOp >>= maybe (error "no optimizer set") pure
      Linear . LinearLayer af op <$> use layerModifierActions

inputs :: forall i n af op m. (KnownNat i, Monad m) => LayerT i n af op m ()
inputs = return ()

type LayerModifierT (n :: Nat) (i :: Nat) m a = StateT [ModifierAction i n] m a

neuronsWith ::
  forall n i af op m.
  (KnownNat n, KnownNat i, Monad m) =>
  LayerModifierT n i m () ->
  LayerT i n af op m ()
neuronsWith modifier = do
  modifierActions <- lift $ execStateT modifier []
  modify $ \s -> s {_layerModifierActions = modifierActions}

neurons :: forall n i af op m. (KnownNat n, KnownNat i, Monad m) => LayerT i n af op m ()
neurons = return ()

weightsScaledBy :: (KnownNat n, KnownNat i, Monad m) => Double -> LayerModifierT n i m ()
weightsScaledBy s = modify (ScaleWeights s :)

biasScaledBy :: (KnownNat n, KnownNat i, Monad m) => Double -> LayerModifierT n i m ()
biasScaledBy s = modify (ScaleBias s :)

activationFunction :: (Monad m) => af -> LayerT i n af op m ()
activationFunction af = modify $ \s -> s {_layerAf = Just af}

optimizerFunction :: (Monad m) => op -> LayerT i n af op m ()
optimizerFunction op = modify $ \s -> s {_layerOp = Just op}
