-- | Any activation function can be implemented outside of this library by
-- defining a data type that implements the `ActivationFunction` and
-- `Differentiable` typeclasses.
module Heuron.V1.Batched.Activation where

class Differentiable a where
  derivative :: a -> (Double -> Double)

class (Differentiable a) => ActivationFunction a where
  activation :: a -> (Double -> Double)

data ReLU = ReLU

instance Differentiable ReLU where
  derivative ReLU x
    | x > 0 = 1
    | otherwise = 0

instance ActivationFunction ReLU where
  activation ReLU = max 0

data Softmax = Softmax

-- TODO: Implement softmax Differentiable.
instance Differentiable Softmax where
  derivative Softmax = undefined

-- TODO: Implement softmax ActivationFunction.
instance ActivationFunction Softmax where
  activation Softmax = undefined
