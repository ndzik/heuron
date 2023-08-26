module Heuron.V2.Layer where

import Control.Lens
import Control.Monad.State
import Data.Default
import Data.Maybe
import GHC.TypeLits

data Layer (i :: Nat) (n :: Nat) af op = Layer af op [ModifierAction i n]

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
  forall i n af op m.
  (KnownNat i, KnownNat n, Monad m) =>
  Int ->
  LayerT i n af op m () ->
  m [Layer i n af op]
mkLayers n = replicateM n . mkLayer

mkLayer :: (KnownNat i, KnownNat n, Monad m) => LayerT i n af op m () -> m (Layer i n af op)
mkLayer builder = evalStateT (builder >> initialize) def
  where
    initialize = do
      af <- use layerAf >>= maybe (error "no activation function set") pure
      op <- use layerOp >>= maybe (error "no optimizer set") pure
      Layer af op <$> use layerModifierActions

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
