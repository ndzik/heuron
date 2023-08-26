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
import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.Functions
import qualified Heuron.V1 as V1
import qualified Heuron.V1.Batched as V1
import qualified Heuron.V1.Batched.Layer.Layer as V1
import Heuron.V2.Backend.Haskell.State
import Heuron.V2.Backend.Translator
import Heuron.V2.Layer
import Heuron.V2.Network

-- | The Haskell backend is a software based backend which uses the CPU to
-- train and infer neural networks.
newtype Haskell a = Haskell {unBackend :: StateT HaskellBackendState IO a}
  deriving (Functor, Applicative, Monad, MonadIO, MonadState HaskellBackendState)

runHaskell :: HaskellBackendState -> Haskell a -> IO a
runHaskell s = flip evalStateT s . unBackend

type family TranslateNetwork n where
  TranslateNetwork (Network b '[]) = V1.Network b '[]
  TranslateNetwork (Network b ls) = V1.Network b (MatchLayers (Network b ls))

type family MatchLayers n where
  MatchLayers (Network b '[]) = '[]
  MatchLayers (Network b (Layer i n af op ': ls)) = V1.Layer b i n af op ': MatchLayers (Network b ls)

type KnownNatConstraint b i i' n n' = (KnownNat b, KnownNat i, KnownNat i', KnownNat n, KnownNat n')

-- Type-Level recursion ends here.
instance
  (KnownNatConstraint b i i' n n') =>
  Translatable Haskell (Network b '[Layer i n af op, Layer i' n' af' op'])
  where
  type
    TargetStructure Haskell (Network b '[Layer i n af op, Layer i' n' af' op']) =
      TranslateNetwork (Network b '[Layer i n af op, Layer i' n' af' op'])

  translate (l1 :=> l2) = do
    v1L1 <- translateLayer @b l1
    v1L2 <- translateLayer @b l2
    pure $ v1L1 V1.:>: v1L2 V1.:>: V1.NetworkEnd
    where
      v1L1 = translateLayer @b l1
      v1L2 = translateLayer @b l2

-- Type-Level recursion starts and continues here.
instance
  (KnownNat b, KnownNat i, KnownNat n, Translatable Haskell (Network b (l1 ': l2 ': ls))) =>
  -- We have to explicitly match the number of layers here, otherwise the
  -- compiler does not know which instance to use.
  Translatable Haskell (Network b (Layer i n af op ': l1 ': l2 ': ls))
  where
  type
    TargetStructure Haskell (Network b (Layer i n af op ': l1 ': l2 ': ls)) =
      TranslateNetwork (Network b (Layer i n af op ': l1 ': l2 ': ls))

  -- Catch end of recurions here.
  translate (l0 :>: l1 :=> ls) = (V1.:>:) <$> translateLayer @b l0 <*> translate (l1 :=> ls)
  -- We also have to explicitly match the number of layers here, otherwise the
  -- instance cannot be resolved for `Translatable Haskell (Network net)`.
  translate (l0 :>: l1 :>: l2 :>: ls) = (V1.:>:) <$> translateLayer @b l0 <*> translate (l1 :>: l2 :>: ls)

translateLayer :: forall b i n af op. (KnownNat i, KnownNat n, KnownNat b) => Layer i n af op -> Haskell (V1.Layer b i n af op)
translateLayer (Layer af op mods) = do
  rng <- use backendRng
  ws <- lift' $ V1.randomMS @n @i rng
  bs <- lift' $ V1.randomVS @n rng
  runModifiers mods $ V1.Layer ws bs zero af op
  where
    lift' = Haskell . lift

runModifiers ::
  (KnownNat b, KnownNat i, KnownNat n) =>
  [ModifierAction i n] ->
  V1.Layer b i n af op ->
  Haskell (V1.Layer b i n af op)
runModifiers [] l = pure l
runModifiers (m : ms) l = runModifier m l >>= runModifiers ms

runModifier ::
  (KnownNat b, KnownNat i, KnownNat n) =>
  ModifierAction i n ->
  V1.Layer b i n af op ->
  Haskell (V1.Layer b i n af op)
runModifier (ScaleWeights s) l = pure $ l {V1._weights = fmap (* s) <$> V1._weights l}
runModifier (ScaleBias s) l = pure $ l {V1._bias = fmap (* s) (V1._bias l)}
