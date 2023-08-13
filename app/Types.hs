{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Types where

import Control.Lens
import GHC.TypeLits (KnownNat, Nat)
import Heuron.V1
import Heuron.V1.Batched
import Heuron.V1.Batched.Network
import Monomer

data HeuronModel (b :: Nat) net = HeuronModel
  { _heuronModelNet :: !(Network b net),
    _heuronModelAvgLoss :: !Double,
    _heuronModelAccuracy :: !Double,
    _heuronModelCurrentEpoch :: !Integer,
    _heuronModelMaxEpochs :: !Integer,
    _heuronModelCurrentBatch :: !(Input b 784 Double)
  }
  deriving (Eq)

makeLenses ''HeuronModel

data UpdateEvent (b :: Nat) net = UpdateEvent
  { _ueAvgBatchLoss :: !Double,
    _ueBatchAccuracy :: !Double,
    _ueEpoch :: !Integer,
    _ueCurrentBatch :: !(Input b 784 Double),
    _ueUpdatedNetwork :: !(Network b net)
  }
  deriving (Show, Eq)

makeLenses ''UpdateEvent

data HeuronEvent (b :: Nat) net
  = HeuronInit
  | HeuronUpdate !(UpdateEvent b net)
  deriving (Show, Eq)
