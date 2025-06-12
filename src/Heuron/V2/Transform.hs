module Heuron.V2.Transform () where

import GHC.TypeLits

data Transform :: * -> * -> * where
  Identity :: Transform a a
  Normalize :: Transform a a
  Compose :: Transform a b -> Transform b c -> Transform a c

infixr 9 :.:

type (:.:) = Compose
