{-# LANGUAGE TypeFamilies #-}

-- | This is the Haskell Heuron interpreter which can be used as an example
-- to see how to implement the interpretation of a neural network described
-- with `Heuron`.
module Heuron.V1.Backend.Haskell.Network where

import Control.Monad.State
import Heuron.Network

type HaskellNetworkM m a = StateT NetworkState m a

type HaskellNetwork a = HaskellNetworkM IO a

data NetworkState

-- | buildNetwork builds a HaskellNetwork from the given Network description.
buildNetwork :: Network -> HaskellNetwork a
buildNetwork (Network i o hs) = undefined
  where
    x = map buildLayer hs
    buildLayer (MkLayer n a) = case a of
      _ -> undefined
