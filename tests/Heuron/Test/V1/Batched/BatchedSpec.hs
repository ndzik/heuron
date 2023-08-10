{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Heuron.Test.V1.Batched.BatchedSpec where

import Data.Functor ((<&>))
import Heuron.Functions
import Heuron.V1
import Heuron.V1.Batched
import System.Random (getStdGen)

batchedSpec :: IO ()
batchedSpec = do
  rng <- getStdGen
  inputLayer <- Layer <$> randomM' @3 @6 rng <*> randomV' @3 rng <*> return zero
  hiddenLayer <- Layer <$> randomM' @3 @3 rng <*> randomV' @3 rng <*> return zero
  outputLayer <- Layer <$> randomM' @2 @3 rng <*> randomV' @2 rng <*> return zero
  let ann =
        inputLayer ReLU (StochasticGradientDescent 1.0)
          :>: hiddenLayer ReLU (StochasticGradientDescent 1.0)
            =| outputLayer Softmax (StochasticGradientDescent 1.0)
      ann' = reverseNetwork ann
  -- Batchsize of 4 with 6 inputs.
  batchedInputs <- mkM' @4 @6 [[1 .. 6] | _ <- [1 .. 4]]
  let (ann', forwardResult) = forward ann batchedInputs
      ann'' = backprop ann forwardResult
  print "batchedSpec:"
  print $ "forwardResult: " <> show forwardResult
  print ann'
