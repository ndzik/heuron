module Heuron.V2.Backend.Haskell.State where

import Control.Lens
import System.Random.Stateful (IOGenM, StdGen)

newtype HaskellBackendState = HaskellBackendState
  { _backendRng :: IOGenM StdGen
  }

makeLenses ''HaskellBackendState
