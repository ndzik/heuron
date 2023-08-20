module Heuron.V2.Layer where

import GHC.TypeLits

data Layer (b :: Nat) (i :: Nat) (n :: Nat) af op = Layer af op
