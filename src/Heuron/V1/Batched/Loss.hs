module Heuron.V1.Batched.Loss where

import Data.Data (Proxy (..))
import Data.Vector (foldl', maxIndex)
import GHC.TypeLits
import Heuron.Functions (mergeEntriesWith)
import Heuron.V1.Batched.Activation (Differentiable (..))
import Heuron.V1.Batched.Input
import Linear (Additive (liftI2))
import Linear.Matrix
import Linear.V
import Linear.Vector ((*^))

class (Differentiable f) => LossComparator f where
  -- | Return the loss function which calculates the loss for a given set of
  -- predictions compared to a set of ground truths. Each entry in the result
  -- corresponds to the loss value for the prediction and truth at the same
  -- index.
  losser :: (KnownNat b, KnownNat n) => f -> (Input b n Double -> Input b n Double -> Input b n Double)

  -- | Return the accuracy function which calculates the accuracy for a given
  -- set of predictions compared to a set of ground truths. The result is a
  -- single value representing the accuracy of the predictions for the given
  -- batch of samples.
  accurator :: (KnownNat b, KnownNat n) => f -> (Input b n Double -> Input b n Double -> Double)

data CategoricalCrossEntropy = CategoricalCrossEntropy deriving (Show, Eq)

instance LossComparator CategoricalCrossEntropy where
  losser CategoricalCrossEntropy = categoricalCrossEntropy
  accurator CategoricalCrossEntropy = categoricalAccuracy

-- | categoricalAccuracy calculates the accuracy for a given set of predictions
-- compared to a set of truth values.
categoricalAccuracy :: forall b n. (KnownNat b, KnownNat n) => Input b n Double -> Input b n Double -> Double
categoricalAccuracy truth prediction =
  let batchSize = fromIntegral $ natVal (Proxy @b)
      correctPredictions = liftI2 matchingPrediction truth prediction
      numOfCorrectPredictions = fromIntegral $ foldr (\p acc -> if p then acc + 1 else acc) 0 correctPredictions
   in numOfCorrectPredictions / batchSize
  where
    matchingPrediction t p = (maxIndex . toVector $ t) == (maxIndex . toVector $ p)

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
categoricalCrossEntropy :: forall b n. (KnownNat b, KnownNat n) => Input b n Double -> Input b n Double -> Input b n Double
categoricalCrossEntropy truth prediction =
  let -- Clip the predicted values to avoid log(0) errors. Adding/Subtracting
      -- tenth of a million to/from the prediction values is a hack but it
      -- should influence the result only very little.
      logPrediction = (log . clip tenthMillion (1 - tenthMillion) <$>) <$> prediction
      losses = mergeEntriesWith (*) truth logPrediction
   in negate <$> losses
  where
    tenthMillion = 1 * 10 ** (-7)
    clip lb ub = max lb . min ub

instance Differentiable CategoricalCrossEntropy where
  derivative CategoricalCrossEntropy = dCategoricalCrossEntropy

-- | The derivative of the categorical cross entropy loss function.
-- ∂L/∂p_i = -t_i/p_i; p = prediction vector, t = truth vector, i = index
dCategoricalCrossEntropy :: forall b n. (KnownNat b, Dim b, Dim n) => V b (V n Double) -> V b (V n Double) -> V b (V n Double)
dCategoricalCrossEntropy truths predictions =
  let samples = fromIntegral $ natVal (Proxy @b)
      gradients = liftI2 (liftI2 (\t p -> if t == 0 || p == 0 then 0 else negate $ t / p)) truths predictions
   in fmap (/ samples) <$> gradients
