{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module V2ExamplePytorch where

import Codec.Serialise
import Control.Lens
import Control.Monad.Except (ExceptT, runExceptT)
import qualified Data.ByteString.Lazy as BSL
import Data.Data (Proxy (..))
import Data.Functor ((<&>))
import Data.Kind (Constraint)
import Digits
import GHC.TypeLits (natVal)
import GHC.TypeNats (KnownNat)
import Heuron.Functions
import Heuron.V1 (mkV)
import Heuron.V1.Batched (CategoricalCrossEntropy (..), network, oneEpoch, runTrainer, trainingResultAccuracy, trainingResultLoss)
import Heuron.V1.Batched.Activation
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Optimizer
import Heuron.V1.Batched.Trainer (TrainerState (..))
import Heuron.V2
import qualified Heuron.V2.Backend as Backend
import qualified Heuron.V2.Backend.Haskell as Backend
import qualified Heuron.V2.Backend.Torch as Torch
import qualified Heuron.V2.Drop as Drop
import qualified Heuron.V2.Residual as Residual
import qualified Heuron.V2.Transform as Transform
import Linear.V
import Monomer
import Streaming (liftIO)
import qualified Streaming as S
import qualified Streaming.Prelude as S
import System.Random (getStdGen, mkStdGen)
import System.Random.Stateful (StateGenM (StateGenM), globalStdGen, newIOGenM)
import Text.Printf (printf)
import Types
import View

generateV2Pytorch :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 16, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 60000) => IO ()
generateV2Pytorch = do
  rng <- newIOGenM (mkStdGen 42069)

  -- Describe network.
  let learningRate = 0.25
  inputLayer <- mkLayer @batchSize $ do
    inputs @'[pixelCount]
    neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 784)
    activationFunction ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  [hiddenLayer00] <- mkLayers 1 $ do
    neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 16)
    activationFunction ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  reshapeLayer <- Transform.mkLayer $ do
    Transform.output @'[28, 28, 28]

  [hiddenLayer01, hiddenLayer02] <- mkLayers 2 $ do
    activationFunction ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  resBlock <- Residual.mkBlock $ do
    Residual.activationFunction ReLU
    Residual.optimizerFunction (StochasticGradientDescent learningRate)
    inputLayer <- mkLayer $ do
      neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 32)
      activationFunction ReLU
      optimizerFunction (StochasticGradientDescent learningRate)

    [hiddenLayer00, hiddenLayer01, hiddenLayer02] <- mkLayers 3 $ do
      neuronsWith @'[hiddenNeuronCount] $ weightsScaledBy (1 / 16)
      activationFunction ReLU
      optimizerFunction (StochasticGradientDescent learningRate)

    dropL <- Drop.mkLayer 0.25

    outputLayer <- mkLayer $ do
      neurons @'[hiddenNeuronCount]
      activationFunction ReLU
      optimizerFunction (StochasticGradientDescent learningRate)

    return $ inputLayer :>: hiddenLayer00 :>: dropL :>: hiddenLayer01 :>: hiddenLayer02 :=> outputLayer

  outputLayer <- mkLayer $ do
    neurons @'[10]
    activationFunction Softmax
    optimizerFunction (StochasticGradientDescent learningRate)

  let ann =
        inputLayer
          :>: reshapeLayer
          :>: hiddenLayer01
          :>: resBlock
          :>: hiddenLayer00
          :=> outputLayer
      code = Torch.runPyTorch ann
  Torch.saveToModule "./torch_module.py" code
  print "PyTorch module written."
