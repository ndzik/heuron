{-# LANGUAGE NumericUnderscores #-}

module Heuron.Test.V1.Batched.TrainerSpec where

import Control.Lens
import Heuron.Functions (mergeEntriesWith, zero)
import Heuron.V1
import Heuron.V1.Batched
import Heuron.V1.Batched.Activation (BackpropableFunction)
import Heuron.V1.Batched.Activation.Function (BackpropableFunction (..))
import System.Random
import Text.Printf (printf)

data Amplifier = Amplifier

instance BackpropableFunction Amplifier where
  backpropagate _ = mergeEntriesWith (*)

instance Differentiable Amplifier where
  derivative _ = derivative ReLU

instance ActivationFunction Amplifier where
  activation _ = (100_000 *) . activation ReLU

trainerSpec :: IO ()
trainerSpec = do
  rng <- getStdGen
  inputLayer <- Layer <$> randomM' @4 @6 rng <*> randomV' @4 rng <*> return zero
  hiddenLayer00 <- Layer <$> randomM' @3 @4 rng <*> randomV' @3 rng <*> return zero
  hiddenLayer01 <- Layer <$> randomM' @3 @3 rng <*> randomV' @3 rng <*> return zero
  outputLayer <- Layer <$> randomM' @3 @3 rng <*> randomV' @3 rng <*> return zero
  let ann =
        inputLayer ReLU (StochasticGradientDescent 1.0)
          :>: hiddenLayer00 ReLU (StochasticGradientDescent 1.0)
          :>: hiddenLayer01 Amplifier (StochasticGradientDescent 1.0)
            =| outputLayer Softmax (StochasticGradientDescent 1.0)
  input <- randomM' @7 @6 rng
  truth <- randomM' @7 @3 rng
  -- TODO: Use a better type error when the construction of the nn is wrong.
  -- Currently we simply get a type error from the compiler, e.g.:
  --
  -- ```
  -- Couldn't match type ‘4’ with ‘7’ arising from a use of ‘oneEpoch’
  -- ```
  --
  -- This happens when a layer with 4 neurons (thus having 4 outputs) is
  -- connected to a layer expecting 7 inputs.
  (accuracy, s) <- runTrainer (oneEpoch input truth) $ TrainerState ann CategoricalCrossEntropy
  -- TODO: Add some visualization helpers for networks.
  print "trainerSpec"
  putStrLn $ printf "Accuracy: %.2f" $ show accuracy
  print "Network:"
  print $ s ^. network
