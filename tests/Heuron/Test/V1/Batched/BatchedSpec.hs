{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Heuron.Test.V1.Batched.BatchedSpec where

import Data.Functor ((<&>))
import Heuron.Functions
import Heuron.V1
import Heuron.V1.Batched

batchedSpec :: IO ()
batchedSpec = do
  inputLayer <- Layer <$> mkM' @3 @6 [[1 .. 6], [1 .. 6], [1 .. 6]] <*> mkV' @3 [1, 2, 3] <*> return zero
  hiddenLayer <- Layer <$> mkM' @3 @3 [(+ 1) <$> [1 .. 3] | _ <- [1 .. 3]] <*> mkV' @3 [1, 2, 3] <*> return zero
  outputLayer <- Layer <$> mkM' @2 @3 [(+ 1) <$> [1 .. 3] | _ <- [1 .. 2]] <*> mkV' @2 [1, 2] <*> return zero
  let ann =
        inputLayer ReLU StochasticGradientDescent
          :>: hiddenLayer ReLU StochasticGradientDescent
            =| outputLayer Softmax StochasticGradientDescent
      ann' = reverseNetwork ann
  -- Batchsize of 4 with 6 inputs.
  batchedInputs <- mkM' @4 @6 [[1 .. 6] | _ <- [1 .. 4]]
  let (ann', forwardResult) = forward ann batchedInputs
      ann'' = backprop ann forwardResult
  print "batchedSpec"
