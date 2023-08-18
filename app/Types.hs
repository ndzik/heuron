{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}

module Types where

import Control.Lens
import Data.Data (Proxy (..))
import Data.Vector
import GHC.TypeLits (KnownNat, Nat)
import GHC.TypeNats
import Heuron.V1
import Heuron.V1.Batched
import Heuron.V1.Batched.Network
import Monomer

data ViewNetwork = ViewNetwork
  { _viewNetworkWeights :: Vector (Vector (Vector Double)),
    _viewNetworkBiases :: Vector (Vector Double),
    _viewNetworkBatchSize :: Int
  }
  deriving (Show, Eq)

data TupleMonoid a b = TupleMonoid a b deriving (Show, Eq)

instance (Semigroup a, Semigroup b) => Semigroup (TupleMonoid a b) where
  (<>) (TupleMonoid a b) (TupleMonoid a' b') = TupleMonoid (a <> a') (b <> b')

instance (Monoid a, Monoid b) => Monoid (TupleMonoid a b) where
  mempty = TupleMonoid mempty mempty

viewNetFromHeuronNet :: forall b net. (KnownNat b, IteratableNetwork (Network b net)) => Network b net -> ViewNetwork
viewNetFromHeuronNet net =
  let (TupleMonoid weights biases) = forLayerIn net $ \(batchSize, inputSize, numOfNeurons) weights biases ->
        TupleMonoid (singleton weights) (singleton biases)
      batchSize = fromIntegral $ natVal (Proxy @b)
   in ViewNetwork
        { _viewNetworkWeights = weights,
          _viewNetworkBiases = biases,
          _viewNetworkBatchSize = batchSize
        }

data HeuronModel = HeuronModel
  { _heuronModelNet :: ViewNetwork,
    _heuronModelAvgLoss :: !Double,
    _heuronModelAccuracy :: !Double,
    _heuronModelCurrentEpoch :: !Integer,
    _heuronModelMaxEpochs :: !Integer
  }
  deriving (Show, Eq)

data UpdateEvent = UpdateEvent
  { _ueAvgBatchLoss :: !Double,
    _ueBatchAccuracy :: !Double,
    _ueEpoch :: !Integer,
    _ueUpdatedNetwork :: ViewNetwork
  }
  deriving (Show, Eq)

data HeuronEvent
  = HeuronInit
  | HeuronUpdate !UpdateEvent
  deriving (Show, Eq)

makeLenses ''ViewNetwork
makeLenses ''HeuronModel
makeLenses ''UpdateEvent
