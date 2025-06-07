module Heuron.V2.Drop where

import GHC.TypeLits
import Heuron.V2.Layer

newtype Drop (b :: Nat) (i :: Nat) = Drop Double

newtype instance Layer (b :: Nat) (i :: Nat) (i :: Nat) (Drop b i) = DropLayer (Drop b i)

mkLayer :: (Monad m, KnownNat b, KnownNat i) => Double -> m (Layer b i i (Drop b i))
mkLayer = return . DropLayer . Drop
