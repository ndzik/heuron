{-# LANGUAGE PolyKinds #-}

module Heuron.V1.Input where

import qualified Data.Vector as V
import Linear.V
import Linear.Vector

-- | Input is the generic input to a neural network with the dimensions nxm.
type Input (n :: k) (m :: k) a = (V n (V m a))
