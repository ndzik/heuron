module Heuron.Test.V1.Batched.LossSpec where

import Heuron.Functions (zero)
import Heuron.V1
import Heuron.V1.Batched
import Heuron.V1.Batched.Loss
import Linear
import Linear.V
import System.Random
import Text.Printf (printf)

lossSpec :: IO ()
lossSpec = do
  let tests =
        [ accuracySpec,
          avgLossSpec
        ]
  sequence_ tests

accuracySpec :: IO ()
accuracySpec = do
  predictions <- mkM' @3 @3 [[0.7, 0.2, 0.1], [0.5, 0.1, 0.4], [0.02, 0.9, 0.08]]
  truths <- mkM' @3 @3 [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
  let accuracy = categoricalAccuracy truths predictions
  -- Accuracy: 0.6666666666666666
  print $ "Accuracy: " <> show accuracy

avgLossSpec :: IO ()
avgLossSpec = do
  predictions <- mkM' @3 @3 [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
  truths <- mkM' @3 @3 [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
  let loss = categoricalCrossEntropy truths predictions
      allLoss = sum . sumV $ loss
      avgLoss = allLoss / 3
  putStrLn . printf "Loss:\n%s" $ prettyMatrix loss
  -- Avg loss: 0.385060880052168
  print $ "Avg loss: " <> show avgLoss

backwardsSpec :: IO ()
backwardsSpec = do
  predictions <- mkM' @3 @3 [[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]]
  truths <- mkM' @3 @3 [[1, 0, 0], [0, 1, 0], [0, 1, 0]]
  return ()
