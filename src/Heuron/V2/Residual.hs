{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.V2.Residual where

import Control.Monad.State
import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.V2.Layer
import qualified Heuron.V2.Network as Network

data Block b ls where
  Block :: Network.Network b ls -> Block b ls

newtype instance Layer (b :: Nat) (i :: Nat) (n :: Nat) (Block b ls) = Residual (Block b ls)

mkBlock ::
  ( KnownNat b,
    KnownNat i,
    KnownNat n,
    Monad m,
    Network.CheckValidLayers (Layer b i n (Block b ls)) (Network.InputLayerOfNetwork ls),
    CheckCompatibleOutput (Layer b i n (Block b ls)) (Network.OutputLayerOfNetwork ls)
  ) =>
  m (Network.Network b ls) ->
  m (Layer b i n (Block b ls))
mkBlock net = Residual . Block <$> net

type CheckCompatibleOutput :: * -> * -> Constraint
type family CheckCompatibleOutput l ls where
  CheckCompatibleOutput (Layer b i n l) (Layer b i' n l') = ()
  CheckCompatibleOutput (Layer b i n l) (Layer b' i' n' l') = TypeError ('Text "Invalid output layer dimension, expecting: " ':<>: 'ShowType n ':<>: 'Text " outputs, but underlying network has output: " ':<>: 'ShowType n')
