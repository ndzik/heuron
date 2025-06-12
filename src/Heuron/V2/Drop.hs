module Heuron.V2.Drop where

import GHC.TypeLits
import Heuron.V2.Layer

newtype DropLayer (b :: Nat) (i :: [Nat]) = DropLayer Double

newtype instance Layer (b :: Nat) (i :: [Nat]) (i :: [Nat]) (DropLayer b i) = Drop (DropLayer b i)

mkLayer :: (Monad m) => Double -> m (Layer b i i (DropLayer b i))
mkLayer = return . Drop . DropLayer
