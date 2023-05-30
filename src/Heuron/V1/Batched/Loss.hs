module Heuron.V1.Batched.Loss where

import GHC.TypeLits
import Heuron.V1.Batched.Input
import Linear.Matrix
import Linear.V

-- | categoricalCrossEntropy (CCE) calculates the loss for a given set of
-- predictions compared to a set of truth values.
--
-- E.g.:
-- > truths <- mkM' @2 @2 [ [1, 0]
--                        , [0, 1]]
-- > predictions <- mkM' @2 @2 [ [0.9, 0.1]
--                             , [0.1, 0.7]]
-- > let result = categoricalCrossEntropy truths predictions
-- > result -- [0.1054, 0.3567]
--
-- Note: The result in the example is truncated for readability.
categoricalCrossEntropy :: (KnownNat n, KnownNat b) => Input n b Double -> Input n b Double -> V n Double
categoricalCrossEntropy truth prediction =
  let predictionT = transpose prediction
      logPredictionT = (log <$>) <$> predictionT
      -- Doing it this way should not lead to unnecessary computation, since
      -- Haskell is lazy and diagnoal will only evaluate the elements in the
      -- diagonal, which corresponds to the calculation done for CCE.
      losses = diagonal $ truth !*! logPredictionT
   in negate <$> losses
