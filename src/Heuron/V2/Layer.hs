module Heuron.V2.Layer where

import Control.Lens
import Control.Monad.State
import Data.Default
import Data.Maybe
import GHC.TypeLits

type LayerSpec = *

data family Layer (b :: Nat) (i :: [Nat]) (o :: [Nat]) (l :: LayerSpec)

data LinearLayer (i :: [Nat]) (o :: [Nat]) af op = LinearLayer af op [ModifierAction i o]

newtype instance Layer (b :: Nat) (i :: [Nat]) (n :: [Nat]) (LinearLayer i n af op) = Linear (LinearLayer i n af op)

type LayerT (i :: [Nat]) (o :: [Nat]) af op m a = StateT (LayerBuilderState i o af op) m a

data ModifierAction (i :: [Nat]) (o :: [Nat]) = ScaleWeights Double | ScaleBias Double

data LayerBuilderState (i :: [Nat]) (o :: [Nat]) af op = LayerBuilderState
  { _layerModifierActions :: ![ModifierAction i o],
    _layerAf :: !(Maybe af),
    _layerOp :: !(Maybe op)
  }

instance Default (LayerBuilderState i n af op) where
  def = LayerBuilderState [] Nothing Nothing

makeLenses ''LayerBuilderState

mkLayers ::
  forall n b af op m.
  (Monad m) =>
  Int ->
  LayerT n n af op m () ->
  m [Layer b n n (LinearLayer n n af op)]
mkLayers n = replicateM n . mkLayer

mkLayer :: forall b i n m af op. (Monad m) => LayerT i n af op m () -> m (Layer b i n (LinearLayer i n af op))
mkLayer builder = evalStateT (builder >> initialize) def
  where
    initialize = do
      af <- use layerAf >>= maybe (error "no activation function set") pure
      op <- use layerOp >>= maybe (error "no optimizer set") pure
      Linear . LinearLayer af op <$> use layerModifierActions

inputs :: forall i n af op m. (Monad m) => LayerT i n af op m ()
inputs = return ()

type LayerModifierT (o :: [Nat]) (i :: [Nat]) m a = StateT [ModifierAction i o] m a

neuronsWith ::
  forall o i af op m.
  (Monad m) =>
  LayerModifierT o i m () ->
  LayerT i o af op m ()
neuronsWith modifier = do
  modifierActions <- lift $ execStateT modifier []
  modify $ \s -> s {_layerModifierActions = modifierActions}

neurons :: forall o i af op m. (Monad m) => LayerT i o af op m ()
neurons = return ()

weightsScaledBy :: (Monad m) => Double -> LayerModifierT o i m ()
weightsScaledBy s = modify (ScaleWeights s :)

biasScaledBy :: (Monad m) => Double -> LayerModifierT o i m ()
biasScaledBy s = modify (ScaleBias s :)

activationFunction :: (Monad m) => af -> LayerT i o af op m ()
activationFunction af = modify $ \s -> s {_layerAf = Just af}

optimizerFunction :: (Monad m) => op -> LayerT i o af op m ()
optimizerFunction op = modify $ \s -> s {_layerOp = Just op}
