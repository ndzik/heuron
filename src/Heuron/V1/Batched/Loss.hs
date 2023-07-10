module Heuron.V1.Batched.Loss where

import GHC.TypeLits
import Heuron.Functions (mergeEntriesWith)
import Heuron.V1.Batched.Activation (Differentiable (..))
import Heuron.V1.Batched.Input
import Linear (Additive (liftI2))
import Linear.Matrix
import Linear.V

class (Differentiable f) => LossComparator f where
  -- | Return the loss function which calculates the loss for a given set of
  -- predictions compared to a set of ground truths. Each entry in the result
  -- corresponds to the loss value for the prediction and truth at the same
  -- index.
  losser :: (KnownNat n, KnownNat b) => f -> (Input n b Double -> Input n b Double -> Input n b Double)

data CategoricalCrossEntropy = CategoricalCrossEntropy

instance LossComparator CategoricalCrossEntropy where
  losser CategoricalCrossEntropy = categoricalCrossEntropy

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
categoricalCrossEntropy :: (KnownNat n, KnownNat b) => Input n b Double -> Input n b Double -> Input n b Double
categoricalCrossEntropy truth prediction =
  let logPrediction = (log <$>) <$> prediction
      -- Doing it this way should not lead to unnecessary computation, since
      -- Haskell is lazy and diagnoal will only evaluate the elements in the
      -- diagonal, which corresponds to the calculation done for CCE.
      losses = mergeEntriesWith (*) truth logPrediction
   in negate <$> losses

instance Differentiable CategoricalCrossEntropy where
  derivative CategoricalCrossEntropy = dCategoricalCrossEntropy

-- | The derivative of the categorical cross entropy loss function.
-- ∂L/∂p_i = -t_i/p_i; p = prediction vector, t = truth vector, i = index
dCategoricalCrossEntropy :: (Dim b, Dim n) => V b (V n Double) -> V b (V n Double) -> V b (V n Double)
dCategoricalCrossEntropy = liftI2 (\t p -> negate <$> (t / p))
