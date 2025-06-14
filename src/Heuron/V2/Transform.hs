module Heuron.V2.Transform where

import Control.Monad.State
import Data.Proxy
import GHC.TypeLits
import Heuron.V2.Layer hiding (mkLayer)

data TransformLayer b i o where
  TransformLayer :: TransformLayer b i o

newtype instance Layer (b :: Nat) (i :: [Nat]) (o :: [Nat]) (TransformLayer b i o) = Transform (TransformLayer b i o)

data TransformState b i o = TransformState

type TransformT (b :: Nat) (i :: [Nat]) (o :: [Nat]) m a = StateT (TransformState b i o) m a

mkLayer :: forall b i o m a. (Monad m) => TransformT b i o m a -> m (Layer b i o (TransformLayer b i o))
mkLayer builder = evalStateT (builder >> return (Transform TransformLayer)) TransformState

output :: forall o b i m. (Monad m) => TransformT b i o m ()
output = return ()

input :: forall i o b m. (Monad m) => TransformT b i o m ()
input = return ()
