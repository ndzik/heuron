{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.V2.Backend.Haskell.Haskell where

import Control.Lens
import Control.Monad.IO.Class
import Control.Monad.RWS
import Control.Monad.State (StateT, evalStateT, lift)
import Data.Default
import Data.Kind (Constraint, Type)
import GHC.TypeLits
import Heuron.Functions
import qualified Heuron.V1 as V1
import qualified Heuron.V1.Batched as V1
import qualified Heuron.V1.Batched.Layer.Layer as V1
import Heuron.V2.Backend.Haskell.State
import Heuron.V2.Backend.Translator
import qualified Heuron.V2.Drop as Drop
import Heuron.V2.Layer
import Heuron.V2.Network
import qualified Heuron.V2.Residual as Residual

-- | The Haskell backend is a software based backend which uses the CPU to
-- train and infer neural networks.
newtype Haskell a = Haskell {unBackend :: StateT HaskellBackendState IO a}
  deriving (Functor, Applicative, Monad, MonadIO, MonadState HaskellBackendState)

runHaskell :: HaskellBackendState -> Haskell a -> IO a
runHaskell s = flip evalStateT s . unBackend

class TranslateLayer b i n l where
  type LayerActivation l :: Type
  type LayerOp l :: Type

  translateLayerImpl :: (i ~ '[i'], n ~ '[n']) => Layer b i n l -> Haskell (V1.Layer b i' n' (LayerActivation l) (LayerOp l))

type family TranslateNetwork n where
  TranslateNetwork (Network b '[]) = V1.Network b '[]
  TranslateNetwork (Network b ls) = V1.Network b (MatchLayers (Network b ls))

type family MatchLayers n where
  MatchLayers (Network b '[]) = '[]
  MatchLayers (Network b (Layer b i n l ': ls)) = V1.Layer b (GetSingleNat i) (GetSingleNat n) (LayerActivation l) (LayerOp l) ': MatchLayers (Network b ls)

type family GetSingleNat n where
  GetSingleNat '[x] = x

instance (i ~ '[i'], n ~ '[n'], KnownNat i', KnownNat n', KnownNat b) => TranslateLayer b i n (LinearLayer i n af op) where
  type LayerActivation (LinearLayer i n af op) = af
  type LayerOp (LinearLayer i n af op) = op

  translateLayerImpl (Linear (LinearLayer af op mods)) = do
    rng <- use backendRng
    ws <- lift' $ V1.randomMS @n' @i' rng
    bs <- lift' $ V1.randomVS @n' rng
    runModifiers mods $ V1.Layer ws bs zero af op
    where
      lift' = Haskell . lift

instance TranslateLayer b i n (Residual.ResidualBlock b ls af op) where
  type LayerActivation (Residual.ResidualBlock b ls af op) = af
  type LayerOp (Residual.ResidualBlock b ls af op) = op
  translateLayerImpl (Residual.Residual (Residual.ResidualBlock net af op)) = do
    undefined

instance (i ~ n) => TranslateLayer b i n (Drop.DropLayer b i) where
  type LayerActivation (Drop.DropLayer b i) = ()
  type LayerOp (Drop.DropLayer b i) = ()
  translateLayerImpl (Drop.Drop (Drop.DropLayer prob)) = do
    undefined

-- Type-Level recursion ends here.
instance
  ( TranslateLayer b i n l,
    TranslateLayer b i' n' l',
    i ~ '[ii],
    i' ~ '[ii'],
    n ~ '[nn],
    n' ~ '[nn']
  ) =>
  Translatable Haskell (Network b '[Layer b i n l, Layer b i' n' l'])
  where
  type
    TargetStructure Haskell (Network b '[Layer b i n l, Layer b i' n' l']) =
      TranslateNetwork (Network b '[Layer b i n l, Layer b i' n' l'])

  translate (l1 :=> l2) = do
    v1L1 <- translateLayer l1
    v1L2 <- translateLayer l2
    let net = v1L1 V1.:>: v1L2 V1.:>: V1.NetworkEnd
    pure net

-- Type-Level recursion starts and continues here.
instance
  ( TranslateLayer b i n l,
    Translatable Haskell (Network b (l1 ': l2 ': ls)),
    i ~ '[i'],
    n ~ '[n']
  ) =>
  -- We have to explicitly match the number of layers here, otherwise the
  -- compiler does not know which instance to use.
  Translatable Haskell (Network b (Layer b i n l ': l1 ': l2 ': ls))
  where
  type
    TargetStructure Haskell (Network b (Layer b i n l ': l1 ': l2 ': ls)) =
      TranslateNetwork (Network b (Layer b i n l ': l1 ': l2 ': ls))

  -- Catch end of recursion here.
  translate (l0 :>: l1 :=> ls) = (V1.:>:) <$> translateLayer l0 <*> translate (l1 :=> ls)
  -- We also have to explicitly match the number of layers here, otherwise the
  -- instance cannot be resolved for `Translatable Haskell (Network net)`.
  translate (l0 :>: l1 :>: l2 :>: ls) = (V1.:>:) <$> translateLayer l0 <*> translate (l1 :>: l2 :>: ls)

translateLayer ::
  forall b i i' n n' l.
  (TranslateLayer b i n l, i ~ '[i'], n ~ '[n']) =>
  Layer b i n l ->
  Haskell (V1.Layer b (GetSingleNat i) (GetSingleNat n) (LayerActivation l) (LayerOp l))
translateLayer = translateLayerImpl

runModifiers ::
  forall b i i' o o' af op.
  (i ~ '[i'], o ~ '[o']) =>
  [ModifierAction i o] ->
  V1.Layer b i' o' af op ->
  Haskell (V1.Layer b i' o' af op)
runModifiers [] l = pure l
runModifiers (m : ms) l = runModifier m l >>= runModifiers ms

runModifier ::
  forall b i i' o o' af op.
  (i ~ '[i'], o ~ '[o']) =>
  ModifierAction i o ->
  V1.Layer b i' o' af op ->
  Haskell (V1.Layer b i' o' af op)
runModifier (ScaleWeights s) l = pure $ l {V1._weights = fmap (* s) <$> V1._weights l}
runModifier (ScaleBias s) l = pure $ l {V1._bias = fmap (* s) (V1._bias l)}
